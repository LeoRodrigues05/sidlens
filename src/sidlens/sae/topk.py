"""TopK sparse autoencoders on residual-stream captures (Gao et al. 2024 recipe).

An SAE rewrites an activation x as a sparse sum of learned directions:

    z   = TopK(W_enc ((x / s) - b_dec) + b_enc)       exactly k non-zeros, all >= 0
    x^  = s * (W_dec z + b_dec)

The scalar s is fixed from the training data so that E||x/s||^2 = d. Latent f
contributes s * z_f * W_dec[f] to x^; that vector is what an ablation removes.

Traps, each with the guard that closes it
-----------------------------------------
1. **A reconstruction splice is not an ablation.** Replacing x by x^ changes
   every latent's worth of reconstruction error at once. `ablate` removes only
   the named latents' contributions from the TRUE x (x - sum s z_f W_dec[f]),
   so the SAE error term stays in place and an empty set is exactly x.
2. **Decoder norm drift.** Unconstrained decoder rows let the encoder shrink
   activations and the decoder grow, so "activation" loses meaning. Rows are
   renormalised to unit norm after every step, and the gradient component
   parallel to each row is removed before the step (Gao et al.).
3. **Dead latents.** TopK SAEs on small corpora leave many latents that never
   fire. The AuxK loss reconstructs the residual error from the k_aux most
   active dead latents. The dead fraction on held-out tokens is reported, never
   hidden.
4. **Reading activations from a different scale.** `s`, `b_dec` and the config
   are saved with the weights. `load` refuses a file whose shapes disagree with
   its config.
5. **Selection on the evaluation data.** Training uses only the rows passed in;
   the experiment passes train-split tokens and evaluates on the test split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SaeConfig:
    d_in: int
    n_latents: int
    k: int
    k_aux: int = 512
    aux_coef: float = 1.0 / 32
    dead_after_tokens: int = 200_000
    lr: float = 2e-4
    batch: int = 4096
    epochs: int = 12
    seed: int = 20260930


def _torch():
    import torch
    return torch


class TopKSAE:
    """Weights held as plain tensors, so the module has no nn.Module state to drift."""

    def __init__(self, cfg: SaeConfig, device="cpu"):
        torch = _torch()
        g = torch.Generator().manual_seed(cfg.seed)
        w = torch.randn(cfg.n_latents, cfg.d_in, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.cfg = cfg
        self.W_dec = w.to(device)                                   # (m, d), unit rows
        self.W_enc = w.clone().to(device)                           # (m, d): z = x W_enc^T
        self.b_enc = torch.zeros(cfg.n_latents, device=device)
        self.b_dec = torch.zeros(cfg.d_in, device=device)
        self.scale = torch.tensor(1.0, device=device)

    # ------------------------------------------------------------- forward --
    def pre(self, x):
        return (x.float() / self.scale - self.b_dec) @ self.W_enc.T + self.b_enc

    def encode(self, x):
        """(idx, val): the k active latents per row, val >= 0."""
        torch = _torch()
        val, idx = torch.topk(self.pre(x), self.cfg.k, dim=-1)
        return idx, torch.relu(val)

    def decode(self, idx, val):
        torch = _torch()
        z = torch.zeros(idx.shape[0], self.cfg.n_latents, device=val.device, dtype=val.dtype)
        z.scatter_(1, idx, val)
        return self.scale * (z @ self.W_dec + self.b_dec)

    def reconstruct(self, x):
        return self.decode(*self.encode(x))

    def contribution(self, idx, val, keep):
        """sum over active latents with keep[row, j] of s * val * W_dec[idx]; (n, d)."""
        v = (val * keep.to(val.dtype)).unsqueeze(-1)                # (n, k, 1)
        return self.scale * (v * self.W_dec[idx]).sum(1)

    def ablate(self, x, latents_per_row):
        """x minus the contributions of the listed latents (error term kept).

        latents_per_row: list of iterables of latent ids, one per row of x.
        Latents that are not active in a row contribute nothing there.
        """
        torch = _torch()
        idx, val = self.encode(x)
        keep = torch.zeros_like(idx, dtype=torch.bool)
        for r, fs in enumerate(latents_per_row):
            fs = torch.as_tensor(sorted(set(int(f) for f in fs)), device=idx.device, dtype=idx.dtype)
            if len(fs):
                keep[r] = torch.isin(idx[r], fs)
        return x.float() - self.contribution(idx, val, keep), keep

    # ---------------------------------------------------------------- train --
    def fit(self, X, log=print, X_val=None):
        """Train on X (n, d) held on the SAE's device (any float dtype). Returns history."""
        torch = _torch()
        cfg = self.cfg
        dev = self.W_dec.device
        n = X.shape[0]
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        with torch.no_grad():
            samp = X[torch.randperm(n, generator=g)[:min(n, 65536)].to(X.device)].float()
            self.scale = (samp.pow(2).sum(1).mean() / cfg.d_in).sqrt()
            self.b_dec = (samp / self.scale).median(0).values       # coordinate-wise median
        params = [self.W_enc, self.b_enc, self.W_dec, self.b_dec]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), eps=6.25e-10)
        steps_per_epoch = max(1, n // cfg.batch)
        total = steps_per_epoch * cfg.epochs
        warm = max(1, total // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * (1.0 if s < 0.8 * total else max(0.0, (total - s) / (0.2 * total))))
        since = torch.zeros(cfg.n_latents, device=dev)              # tokens since last fired
        hist, step = [], 0
        for ep in range(cfg.epochs):
            perm = torch.randperm(n, generator=g)
            for b in range(steps_per_epoch):
                xb = X[perm[b * cfg.batch:(b + 1) * cfg.batch].to(X.device)].to(dev).float() / self.scale
                pre = (xb - self.b_dec) @ self.W_enc.T + self.b_enc
                val, idx = torch.topk(pre, cfg.k, dim=-1)
                val = torch.relu(val)
                z = torch.zeros_like(pre).scatter(1, idx, val)
                xh = z @ self.W_dec + self.b_dec
                err = xb - xh
                mse = err.pow(2).sum(1).mean()
                var = (xb - xb.mean(0)).pow(2).sum(1).mean()
                with torch.no_grad():
                    fired = torch.zeros(cfg.n_latents, device=dev, dtype=torch.bool)
                    fired[idx[val > 0].unique()] = True
                    since = torch.where(fired, torch.zeros_like(since), since + xb.shape[0])
                    dead = since > cfg.dead_after_tokens
                aux = torch.zeros((), device=dev)
                if dead.any():
                    ka = min(cfg.k_aux, int(dead.sum()))
                    pre_d = pre.masked_fill(~dead, float("-inf"))
                    va, ia = torch.topk(pre_d, ka, dim=-1)
                    za = torch.zeros_like(pre).scatter(1, ia, torch.relu(va))
                    ea = za @ self.W_dec
                    aux = (err.detach() - ea).pow(2).sum(1).mean() / var.detach()
                loss = mse / var + cfg.aux_coef * aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():                               # trap 2: keep rows unit-norm
                    gw = self.W_dec.grad
                    gw -= (gw * self.W_dec).sum(1, keepdim=True) * self.W_dec
                opt.step()
                sched.step()
                with torch.no_grad():
                    self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)
                step += 1
                if step % 200 == 0 or step == total:
                    rec = {"step": step, "epoch": ep, "fvu": float(mse / var), "aux": float(aux),
                           "dead_frac": float(dead.float().mean()), "lr": float(sched.get_last_lr()[0])}
                    if X_val is not None:
                        rec["fvu_val"] = self.fvu(X_val)
                    hist.append(rec)
                    log(json.dumps(rec))
        for p in params:
            p.requires_grad_(False)
        return hist

    def fvu(self, X, batch: int = 16384) -> float:
        """Fraction of variance unexplained on X (n, d), in the original scale."""
        torch = _torch()
        with torch.no_grad():
            dev = self.W_dec.device
            mu = torch.zeros(self.cfg.d_in, device=dev, dtype=torch.float64)
            n = X.shape[0]
            for b in range(0, n, batch):
                mu += X[b:b + batch].to(dev).double().sum(0)
            mu /= n
            se = sv = 0.0
            for b in range(0, n, batch):
                xb = X[b:b + batch].to(dev).float()
                se += float((xb - self.reconstruct(xb)).double().pow(2).sum())
                sv += float((xb.double() - mu).pow(2).sum())
        return se / sv

    # ------------------------------------------------------------------- io --
    def save(self, path: Path) -> None:
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        save_file({"W_enc": self.W_enc.detach().cpu().contiguous(), "W_dec": self.W_dec.detach().cpu().contiguous(),
                   "b_enc": self.b_enc.detach().cpu(), "b_dec": self.b_dec.detach().cpu(),
                   "scale": self.scale.detach().cpu().reshape(1)}, str(path / "sae.safetensors"))
        (path / "config.json").write_text(json.dumps(asdict(self.cfg), indent=1))

    @classmethod
    def load(cls, path: Path, device="cpu") -> "TopKSAE":
        from safetensors.torch import load_file
        path = Path(path)
        cfg = SaeConfig(**json.loads((path / "config.json").read_text()))
        t = load_file(str(path / "sae.safetensors"))
        want = {"W_enc": (cfg.n_latents, cfg.d_in), "W_dec": (cfg.n_latents, cfg.d_in),
                "b_enc": (cfg.n_latents,), "b_dec": (cfg.d_in,), "scale": (1,)}
        for k, s in want.items():
            if tuple(t[k].shape) != s:
                raise ValueError(f"{path}: {k} has shape {tuple(t[k].shape)}, config says {s}")
        sae = cls(cfg, device="cpu")
        sae.W_enc, sae.W_dec = t["W_enc"].to(device), t["W_dec"].to(device)
        sae.b_enc, sae.b_dec = t["b_enc"].to(device), t["b_dec"].to(device)
        sae.scale = t["scale"][0].to(device)
        return sae
