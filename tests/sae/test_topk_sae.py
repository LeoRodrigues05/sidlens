"""TopK SAE: exact sparsity, unit decoder rows, error-preserving ablation, io round trip.

CPU, toy sizes. The ablation identity is the property the causal SAE runs
rely on: removing no latent must return the input bit for bit, and removing
every active latent must leave exactly the reconstruction error plus b_dec.
"""

import pytest

torch = pytest.importorskip("torch")

from sidlens.sae.topk import SaeConfig, TopKSAE  # noqa: E402


def toy(n=4096, d=16, m=48, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    atoms = torch.randn(m, d, generator=g)
    idx = torch.randint(0, m, (n, k), generator=g)
    coef = torch.rand(n, k, generator=g) * 3
    return (coef.unsqueeze(-1) * atoms[idx]).sum(1) + 5.0


def test_encode_is_exactly_k_sparse():
    sae = TopKSAE(SaeConfig(d_in=16, n_latents=64, k=5))
    idx, val = sae.encode(torch.randn(10, 16))
    assert idx.shape == (10, 5) and val.shape == (10, 5) and bool((val >= 0).all())


def test_training_reduces_fvu_and_keeps_unit_decoder():
    X = toy()
    sae = TopKSAE(SaeConfig(d_in=16, n_latents=64, k=4, batch=512, epochs=6, lr=3e-3,
                            dead_after_tokens=2048, k_aux=16))
    before = sae.fvu(X)
    sae.fit(X, log=lambda *_: None)
    after = sae.fvu(X)
    assert after < 0.5 * before
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_ablation_identity_and_complement():
    X = toy(n=64)
    sae = TopKSAE(SaeConfig(d_in=16, n_latents=64, k=4, batch=64, epochs=2, lr=3e-3))
    sae.fit(toy(), log=lambda *_: None)
    none, keep = sae.ablate(X, [[] for _ in range(len(X))])
    assert torch.equal(none, X.float()) and not keep.any()
    idx, _ = sae.encode(X)
    allf, keep = sae.ablate(X, [r.tolist() for r in idx])
    assert keep.all()
    want = X.float() - sae.reconstruct(X) + sae.scale * sae.b_dec
    assert torch.allclose(allf, want, atol=1e-4)
    inactive = [[int(f) for f in range(64) if f not in set(r.tolist())][:3] for r in idx]
    same, keep = sae.ablate(X, inactive)
    assert torch.equal(same, X.float()) and not keep.any()


def test_save_load_round_trip(tmp_path):
    sae = TopKSAE(SaeConfig(d_in=8, n_latents=16, k=2))
    sae.scale = torch.tensor(2.5)
    sae.save(tmp_path / "s")
    back = TopKSAE.load(tmp_path / "s")
    x = torch.randn(5, 8)
    assert torch.equal(sae.reconstruct(x), back.reconstruct(x))
    with pytest.raises(FileExistsError):
        sae.save(tmp_path / "s")
