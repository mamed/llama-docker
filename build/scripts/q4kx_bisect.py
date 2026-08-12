#!/usr/bin/env python3
"""
Q4_K_XL GPU-layer bisect probe for Qwen3.5 AgentWorld 35B-A3B.

Usage:  python3 q4kx_bisect.py <K> [<K> ...]

For each K: spin up the probe container, wait for /health, capture stats,
run a single PONG completion, tear down. Reports OOM immediately on failure.
"""
import json
import os
import shlex
import subprocess
import sys
import time

CONTAINER = "q4kx-probe"
PROBE_PORT = 64060
SERVER_PORT = 64006
HEALTH_TIMEOUT = 70

BASE_OVERRIDE_FMT = (
    "blk\\.0\\.ffn_.*_exps=CPU,"
    "blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
    "blk\\.2[{cpu24}]\\.ffn_.*_exps=CPU,"
    "blk\\.3[0-9]\\.ffn_.*_exps=CPU,"
    "blk\\.4[0-9]\\.ffn_.*_exps=CPU"
)


def override_for_k(k: int) -> tuple[str, str]:
    """Return (override_string, gpu_range_str) for K GPU expert blocks (20..20+K-1)."""
    last = 20 + k - 1
    gpu_range = f"20-{last}"
    # CPU patterns: all blocks EXCEPT 20..20+K-1 must be on CPU.
    # 40 layers total, 0..39. Keep all on CPU by listing them explicitly,
    # then let -ngl 99 (with no explicit GPU rule) place the rest on GPU.
    # But the working v5 IQ4_NL spec only listed CPU patterns, relying on
    # -ngl 99 + "everything not on CPU" implicitly on GPU. Mirror that.
    # For K=4: GPU=20-23, CPU=0,10-19,24-39
    if k == 4:
        cpu_high24 = "4-9"
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.2[4-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.3[0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    if k == 5:
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.2[5-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.3[0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    if k == 6:
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.2[6-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.3[0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    if k == 7:
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.2[7-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.3[0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    if k == 8:
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.2[8-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.3[0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    if k == 9:
        # Replicate the v5 IQ4_NL working pattern exactly (only blk.29 forced to CPU)
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.2[9]\\.ffn_.*_exps=CPU,"
            f"blk\\.[3-4][0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    if k == 10:
        return (
            f"blk\\.0\\.ffn_.*_exps=CPU,"
            f"blk\\.1[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.3[0-9]\\.ffn_.*_exps=CPU,"
            f"blk\\.4[0-9]\\.ffn_.*_exps=CPU"
        ), gpu_range
    raise ValueError(f"Unsupported K={k}")


def sh(cmd: str) -> tuple[int, str]:
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def wait_health(label: str) -> bool:
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        rc, out = sh(f"curl -sf http://127.0.0.1:{PROBE_PORT}/health")
        if rc == 0 and '"status":"ok"' in out:
            print(f"  [{label}] /health OK after {HEALTH_TIMEOUT - (deadline - time.time()):.1f}s")
            return True
        # Check if container has died
        rc2, _ = sh(f"docker ps --format '{{{{.Names}}}}' | grep -q '^{CONTAINER}$'")
        if rc2 != 0:
            print(f"  [{label}] container exited during health wait")
            return False
        time.sleep(1)
    print(f"  [{label}] /health did not respond in {HEALTH_TIMEOUT}s")
    return False


def collect_logs_tail(n: int = 14) -> str:
    rc, out = sh(f"docker logs --tail {n} {CONTAINER} 2>&1")
    return out


def props_n_ctx() -> tuple[int | None, int | None]:
    rc, out = sh(f"curl -sf http://127.0.0.1:{PROBE_PORT}/props")
    if rc != 0:
        return None, None
    p = json.loads(out)
    dgs = p.get("default_generation_settings", {}) or {}
    return dgs.get("n_ctx"), p.get("n_ctx")


def vram_mib() -> tuple[int, int]:
    rc, out = sh(
        "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits"
    )
    if rc != 0:
        return -1, -1
    parts = [p.strip() for p in out.strip().split(",")]
    return int(parts[0]), int(parts[1])


def pong_probe() -> dict:
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
        "max_tokens": 10,
        "stream": False,
    }
    body_json = json.dumps(body)
    t0 = time.time()
    rc, out = sh(
        f"curl -sf http://127.0.0.1:{PROBE_PORT}/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d {shlex.quote(body_json)}"
    )
    t1 = time.time()
    wall = t1 - t0
    if rc != 0:
        return {"ok": False, "raw": out, "wall_s": wall}
    try:
        r = json.loads(out)
    except json.JSONDecodeError:
        return {"ok": False, "raw": out, "wall_s": wall}
    u = r.get("usage", {}) or {}
    content = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
    pt = u.get("prompt_tokens", 0) or 0
    ct = u.get("completion_tokens", 0) or 0
    gen_tps = ct / wall if wall > 0 else 0
    prompt_tps = pt / wall if wall > 0 else 0
    return {
        "ok": True,
        "raw": r,
        "content": content,
        "usage": u,
        "wall_s": wall,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
    }


def check_oom_in_logs() -> str | None:
    """Return the OOM/failure line if found in recent logs, else None."""
    rc, out = sh(
        f"docker logs --tail 60 {CONTAINER} 2>&1 | "
        f"grep -E -i 'out of memory|allocat|kv cache|failed to allocate|graph_reserve|error' | head -5"
    )
    if rc == 0 and out.strip():
        return out.strip()
    return None


def gpu_precheck(label: str, expected_free_min_mib: int = 500) -> tuple[int, int, str]:
    """Verify the GPU is actually free of competing compute apps. Return (used, free, note)."""
    rc, out = sh(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    rc2, out2 = sh(
        "nvidia-smi --query-gpu=memory.used,memory.free "
        "--format=csv,noheader,nounits"
    )
    used = int(out2.strip().split(",")[1].strip())
    free = int(out2.strip().split(",")[0].strip()) if False else 0
    # Parse "used, free" — column order is memory.used, memory.free
    parts = [p.strip() for p in out2.strip().split(",")]
    used = int(parts[0])
    free = int(parts[1])
    competing = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        comps = [c.strip() for c in line.split(",")]
        try:
            pid = int(comps[0])
        except ValueError:
            continue
        proc = comps[1] if len(comps) > 1 else ""
        mem = int(comps[2]) if len(comps) > 2 and comps[2].isdigit() else 0
        competing.append((pid, proc, mem))
    note = (
        f"used={used} MiB free={free} MiB "
        f"competing_apps={[(p,n,m) for p,n,m in competing]}"
    )
    print(f"  [{label}] PRECHECK: {note}")
    return used, free, note


def run_probe(k: int) -> dict:
    override, gpu_range = override_for_k(k)
    result = {
        "k": k,
        "gpu_range": gpu_range,
        "override": override,
        "healthy": False,
        "oom": None,
        "n_ctx": None,
        "vram_used": None,
        "vram_free": None,
        "precheck_note": None,
        "pong": None,
    }

    print(f"\n===== K={k} (GPU={gpu_range}) =====")
    print(f"override-tensor: {override}")

    sh(f"docker rm -f {CONTAINER} 2>/dev/null || true")
    time.sleep(2)

    # PRECHECK: there must be no competing llama-server holding VRAM. We don't
    # kill prod containers here — we just report and proceed only if GPU is
    # free enough for a probe to be representative.
    used0, free0, note = gpu_precheck(f"K={k}")
    result["precheck_note"] = note
    if free0 < 200:
        print(f"  [{k}] GPU looks too busy to bisect reliably: free={free0} MiB")
        result["oom"] = f"precheck: only {free0} MiB free; competing compute apps present"
        return result

    run_cmd = (
        f"docker run -d --name {CONTAINER} --gpus all "
        f'-v "/run/media/fekry/OS shared/LLM-Models:/models:ro" '
        f"-p {PROBE_PORT}:{SERVER_PORT} "
        f"-e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility "
        f"--entrypoint /usr/local/bin/llama-server "
        f"llama-qwen-35b-agentworld:latest "
        f"-m /models/unsloth/Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL/Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL.gguf "
        f"--host 0.0.0.0 --port {SERVER_PORT} "
        f"-ngl 99 --ctx-size 262144 "
        f"-ctk q4_0 -ctv q4_0 "
        f"--parallel 1 --cont-batching "
        f"-fa on --no-mmap -t 16 -b 2048 "
        f"--override-tensor {shlex.quote(override)} "
        f"--chat-template-file /usr/local/share/llama-server/qwen35-permissive.jinja "
        f"--reasoning-budget -1 --agent"
    )
    rc, out = sh(run_cmd)
    print(f"  docker run rc={rc}: {(out or '').strip()[:200]}")
    if rc != 0:
        result["oom"] = f"docker run failed: {out.strip()}"
        return result

    healthy = wait_health(f"K={k}")
    if not healthy:
        result["oom"] = check_oom_in_logs() or "container died before /health"
        return result

    result["healthy"] = True
    n_default, n_max = props_n_ctx()
    result["n_ctx"] = n_default
    print(f"  n_ctx = {n_default} (max={n_max})")
    print("  last 14 log lines:")
    for line in collect_logs_tail(14).splitlines():
        print(f"    {line}")
    used, free = vram_mib()
    result["vram_used"] = used
    result["vram_free"] = free
    print(f"  VRAM used={used} MiB  free={free} MiB")

    pong = pong_probe()
    result["pong"] = pong
    if pong.get("ok"):
        print(f"  PONG reply: {pong['content']!r}")
        print(
            f"  prompt_tokens={pong['usage'].get('prompt_tokens')} "
            f"completion_tokens={pong['usage'].get('completion_tokens')} "
            f"wall={pong['wall_s']:.3f}s"
        )
        print(f"  prompt_t/s={pong['prompt_tps']:.2f}  gen_t/s={pong['gen_tps']:.2f}")
    else:
        print(f"  PONG failed: rc={pong.get('raw','')[:200]}")

    return result


def teardown():
    sh(f"docker rm -f {CONTAINER} 2>/dev/null || true")
    time.sleep(2)


def main():
    ks = [int(x) for x in sys.argv[1:]]
    if not ks:
        ks = [4, 5, 6, 7, 8]
    rows = []
    oom_k: int | None = None
    for k in ks:
        r = run_probe(k)
        teardown()
        rows.append(r)
        if not r["healthy"]:
            oom_k = k
            print(f"\n>>> FOR Q4_K_XL: K={k} BLOCKS OOMs <<<")
            break
    out = {
        "ks_tested": [r["k"] for r in rows],
        "oom_k": oom_k,
        "results": rows,
    }
    print("\n===== JSON-RESULTS =====")
    print(json.dumps(out, indent=2, default=str))
    # Persist a copy for the parent agent.
    with open("/home/fekry/llama-docker/build/scripts/q4kx_bisect_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
