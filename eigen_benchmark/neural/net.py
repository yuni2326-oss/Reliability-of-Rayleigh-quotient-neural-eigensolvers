"""시드앙상블 MLP — 하드 경계조건 φ(x) = x²·N(x).

x² 인자가 φ(0) = φ′(0) = 0을 **구조적으로** 만족시키므로 약형식에 벌점이 필요 없다.
`torch.func.stack_module_state`로 시드별 파라미터를 하나의 배치 텐서로 쌓아
vmap 한 번에 전 시드를 전진·역전파한다 — 순차 루프 대비 수십 배(스파이크: 50시드
4000 iter이 스테이지당 ~134 s).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import stack_module_state


class _MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return (x ** 2) * self.net(x)


class EnsembleMLP:
    """n_seeds개 독립 초기화를 스택한 앙상블."""

    def __init__(self, n_seeds: int, width: int = 64, depth: int = 4,
                 seed: int = 0, device: str | None = None, dtype=None):
        dtype = torch.float64 if dtype is None else dtype
        # 축소정밀도는 기본 dtype으로 거부될 수 있다 — 그때는 fp32를 기본으로 두고
        # 파라미터만 목표 dtype으로 내린다(연산은 파라미터 dtype을 따른다).
        try:
            torch.set_default_dtype(dtype)
        except TypeError:
            torch.set_default_dtype(torch.float32)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        models = [_MLP(width, depth).to(dev).to(dtype) for _ in range(n_seeds)]
        self.params, _ = stack_module_state(models)
        self.base = _MLP(width, depth).to(dev).to(dtype).to("meta")
        self.n_seeds, self.device, self.dtype = n_seeds, dev, dtype
        self.width, self.depth, self.seed = width, depth, seed

    def clone_params(self):
        return {k: v.detach().clone() for k, v in self.params.items()}

    def disclosure(self) -> dict:
        """Appendix B용 구조 공개."""
        n = sum(v[0].numel() for v in self.params.values())
        return {"width": self.width, "depth": self.depth, "activation": "tanh",
                "hard_bc": "phi = x^2 * N(x)", "n_params_per_seed": int(n),
                "n_seeds": self.n_seeds,
                "dtype": str(getattr(self, "dtype", torch.float64)).replace("torch.", ""),
                "seed": self.seed}
