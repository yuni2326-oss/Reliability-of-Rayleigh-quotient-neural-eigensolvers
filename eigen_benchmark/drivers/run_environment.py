"""실행 환경 기록 — data/code availability 절이 손으로 쓰이지 않게 한다.

두 환경에서 각각 돌린다. 고전 풀이는 호스트 CPU에서, 신경 arm은 컨테이너 GPU에서
돌기 때문에 하나의 기록으로는 둘을 다 말할 수 없다.

  # 호스트(고전·mpmath·렌더)
  PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python \\
      -m eigen_benchmark.drivers.run_environment

  # 컨테이너(신경 arm)
  docker run --rm --gpus all -v /home/super/project:/w -w /w -e PYTHONPATH=/w \\
      --entrypoint bash gb10-pinn:26.04 -lc \\
      'python -m eigen_benchmark.drivers.run_environment'

`torch`가 보이면 `environment_gpu.json`, 아니면 `environment_host.json`으로 쓴다.
**측정값만 기록한다** — 없는 것은 넣지 않고, 렌더러가 빈 칸을 그대로 드러낸다.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from . import manifest


def _run(*cmd) -> str:
    try:
        env = dict(os.environ, LC_ALL="C")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                              env=env).stdout
    except Exception:
        return ""


def cpu_model() -> str:
    """aarch64에는 /proc/cpuinfo의 model name이 없다 — lscpu를 C 로케일로 읽는다.

    GB10처럼 큰/작은 코어가 섞인 칩은 `Model name`이 클러스터마다 나오므로 전부 모아
    코어 수와 함께 적는다. 하나만 집으면 절반을 숨기게 된다."""
    out = _run("lscpu")
    names, cores = [], []
    for ln in out.splitlines():
        if ln.startswith("Model name:"):
            names.append(ln.split(":", 1)[1].strip())
        elif ln.startswith("Core(s) per socket:"):
            cores.append(ln.split(":", 1)[1].strip())
    if not names:
        for ln in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if ln.lower().startswith("model name"):
                names.append(ln.split(":", 1)[1].strip())
                break
    n = _run("nproc").strip()
    if len(names) > 1 and len(cores) == len(names):
        parts = " + ".join(f"{c}× {nm}" for nm, c in zip(names, cores))
        return f"{parts} ({n} threads total)"
    return f"{names[0]} ({n} threads)" if names else "unknown"


def gpu_model() -> dict:
    out = _run("nvidia-smi", "--query-gpu=name,driver_version",
               "--format=csv,noheader")
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return {}
    parts = [p.strip() for p in line.split(",")]
    d = {"gpu": parts[0]}
    if len(parts) > 1:
        d["nvidia_driver"] = parts[1]
    return d


def os_release() -> str:
    for ln in _run("cat", "/etc/os-release").splitlines():
        if ln.startswith("PRETTY_NAME="):
            return ln.split("=", 1)[1].strip().strip('"')
    return platform.platform()


def collect() -> dict:
    rec = manifest.build({"record": "execution environment"})
    rec["cpu"] = cpu_model()
    rec["os"] = os_release()
    rec["kernel"] = platform.release()
    rec.update(gpu_model())
    try:
        import matplotlib
        rec["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    try:
        import torch
        rec["role"] = "gradient-trained and random-feature arms (GPU)"
        rec["torch"] = torch.__version__
        rec["torch_cuda"] = torch.version.cuda or "cpu build"
        cud = torch.backends.cudnn.version()
        if cud:
            rec["cudnn"] = str(cud)
        # **fp64는 환경 기본값이 아니다.** 프로세스 기본은 float32이고 솔버가 진입 시
        # 명시적으로 float64로 바꾼다. 그 사실을 그대로 기록한다.
        rec["torch_default_dtype_at_import"] = str(torch.get_default_dtype())
        rec["fp64"] = ("solvers call torch.set_default_dtype(torch.float64) at "
                       "entry; process default is float32")
        tag = "gpu"
    except Exception:
        rec["role"] = ("classical solves, high-precision references, render "
                       "pipeline (CPU)")
        rec["fp64"] = "numpy float64 throughout; mpmath dps=50 for references"
        tag = "host"
    rec["python_executable"] = sys.executable
    return rec, tag


def main(outdir=None) -> dict:
    d = manifest.ensure_outdir(outdir)
    rec, tag = collect()
    p = os.path.join(d, f"environment_{tag}.json")
    manifest.write_json(p, rec)
    print(f"[run_environment] {p} — {rec.get('role')}")
    return rec


if __name__ == "__main__":
    main()
