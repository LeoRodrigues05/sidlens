"""Length buckets in `FixedShapeRunner`: a row's scores depend on the row alone.

Tiny random Qwen2 with the frozen RQ-KMeans 3x128 tokenizer, CPU. Histories of
1 to 40 items put rows in several buckets.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sidlens import paths  # noqa: E402
from sidlens.data import ar_prompts as P  # noqa: E402
from sidlens.data.sids import SidTable  # noqa: E402
from sidlens.interventions.runner import FixedShapeRunner, Job, length_buckets  # noqa: E402
from sidlens.interventions.scoring import DigitScorer  # noqa: E402
from sidlens.models import ar  # noqa: E402

CKPT = "next-item_best"


@pytest.fixture(scope="module")
def setup():
    if not (paths.FROZEN_CKPT / "ar" / CKPT / "tokenizer.json").exists():
        pytest.skip("frozen AR tokenizer not on this machine")
    tok = transformers.AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / CKPT), local_files_only=True)
    vocab = ar.load_vocab(CKPT)
    table = SidTable.load(vocab.variant)
    sids = sorted({"".join(f"<{'abc'[d]}_{c}>" for d, c in enumerate(code)) for code in table.cb2items})
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    encs = []
    for i, n in enumerate([1, 2, 3, 12, 20, 40, 5, 33]):
        h = tuple(sids[(7 * i + k) % len(sids)] for k in range(n))
        ex = P.ArExample(f"e{i}", "next-item", "test", vocab.variant.name, i, f"u{i}", tuple(range(n)), h,
                         (0,), (sids[i],))
        encs.append(P.encode(ex, tok, vocab, template="v3", with_target=True))
    return model, tok, DigitScorer(vocab, table), encs


def _run(runner, encs, order):
    s, cl, _, _ = runner.forward([Job(encs[i]) for i in order])
    return {i: (s["logp_codes"][r].tolist(), [c[r] for c in cl]) for r, i in enumerate(order)}


def test_rows_do_not_depend_on_their_batch(setup):
    model, tok, scorer, encs = setup
    lens = [len(e) for e in encs]
    buckets = length_buckets(lens)
    assert len(buckets) >= 3
    t_pad = buckets[-1]
    r = FixedShapeRunner(model, scorer, tok.eos_token_id, t_pad, 4, 1, buckets=buckets)
    mixed = _run(r, encs, [0, 5, 3, 7])           # four buckets in one call -> split
    alone = {}
    for i in (0, 5, 3, 7):
        alone.update(_run(r, encs, [i]))
    for i in mixed:
        assert mixed[i][0] == alone[i][0]
        assert all(torch.equal(a, b) for a, b in zip(mixed[i][1], alone[i][1]))
    # in one bucket, a bucketed runner equals an unbucketed runner padded to that bucket
    b0 = r.bucket_len(lens[0])
    plain = FixedShapeRunner(model, scorer, tok.eos_token_id, b0, 4, 1)
    same = [i for i, n in enumerate(lens) if r.bucket_len(n) == b0]
    a, b = _run(plain, encs, same), _run(r, encs, same)
    for i in same:
        assert a[i][0] == b[i][0]
        assert all(torch.equal(x, y) for x, y in zip(a[i][1], b[i][1]))


def test_batches_stay_in_one_bucket_and_cover_every_row(setup):
    model, tok, scorer, encs = setup
    lens = [len(e) for e in encs]
    r = FixedShapeRunner(model, scorer, tok.eos_token_id, length_buckets(lens)[-1], 3, 1,
                         buckets=length_buckets(lens))
    chunks = r.batches(range(len(encs)), lambda i: lens[i])
    assert sorted(i for c in chunks for i in c) == list(range(len(encs)))
    assert all(len({r.bucket_len(lens[i]) for i in c}) == 1 and len(c) <= 3 for c in chunks)
    plain = FixedShapeRunner(model, scorer, tok.eos_token_id, length_buckets(lens)[-1], 3, 1)
    assert plain.batches(range(8), lambda i: lens[i]) == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_capture_across_buckets_is_refused(setup):
    model, tok, scorer, encs = setup
    lens = [len(e) for e in encs]
    r = FixedShapeRunner(model, scorer, tok.eos_token_id, length_buckets(lens)[-1], 4, 1,
                         buckets=length_buckets(lens))
    with pytest.raises(ValueError, match="one length bucket"):
        r.forward([Job(encs[0]), Job(encs[5])], capture=["model.layers.0"])


def test_every_row_keeps_a_pad_token(setup):
    """A row filling its bucket would give a forward with no padding, which takes a different
    attention kernel; buckets are strictly longer than their rows."""
    assert length_buckets([32, 33, 64, 10]) == [32, 64, 96]
    model, tok, scorer, _ = setup
    r = FixedShapeRunner(model, scorer, tok.eos_token_id, 96, 4, 1, buckets=length_buckets([32, 33, 64, 10]))
    assert [r.bucket_len(n) for n in (10, 31, 32, 33, 63, 64)] == [32, 32, 64, 64, 64, 96]
