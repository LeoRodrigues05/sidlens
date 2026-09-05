"""The capture layer: ordering, cleanup, and the tuple-output hazard.

Built on toy modules rather than the real checkpoints so it runs in a second.
The behaviours under test are structural -- which modules are read, in what
order, whether hooks are removed -- and none of them need 3 GB of Qwen weights
to go wrong. The real checkpoints are exercised by `scripts/hooks_smoke.py`
under SLURM, because loading them on a login node takes tens of minutes of
disk wait.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from sidlens import hooks as H  # noqa: E402


class Block(nn.Module):
    """A block that returns a tuple, like every real transformer block."""

    def __init__(self, d=4):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x), None       # attention-weights slot, empty here


class BareBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class TwoStack(nn.Module):
    """An encoder stack and a decoder stack, as DIFF_GRM has."""

    def __init__(self, n_enc=2, n_dec=3, d=4):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(Block(d) for _ in range(n_enc))
        self.decoder_blocks = nn.ModuleList(Block(d) for _ in range(n_dec))

    def forward(self, x):
        for b in self.encoder_blocks:
            x = b(x)[0]
        for b in self.decoder_blocks:
            x = b(x)[0]
        return x


@pytest.fixture
def stack():
    m = TwoStack()
    H.RESIDUAL_PATTERNS["TwoStack"] = (r"^encoder_blocks\.\d+$",
                                       r"^decoder_blocks\.\d+$")
    yield m
    H.RESIDUAL_PATTERNS.pop("TwoStack", None)


def test_encoder_is_read_before_decoder(stack):
    """The bug this pins: sorting read positions by layer NUMBER put
    `decoder_blocks.0` ahead of `encoder_blocks.0`, so a residual stream would
    be read with the decoder before the encoder that feeds it."""
    assert H.residual_points(stack) == [
        "encoder_blocks.0", "encoder_blocks.1",
        "decoder_blocks.0", "decoder_blocks.1", "decoder_blocks.2"]


def test_layers_beyond_nine_sort_numerically():
    """`layers.10` must not sort between `layers.1` and `layers.2`."""
    names = [f"model.layers.{i}" for i in (0, 1, 2, 9, 10, 11, 21)]
    assert sorted(names, key=H._layer_index) == names


def test_capture_records_which_tuple_element_it_took(stack):
    _, cap = H.run_capture(stack, lambda: stack(torch.randn(2, 4)))
    assert len(cap) == 5
    for c in cap.data.values():
        assert c.took_index == 0
        assert tuple(c.shape) == (2, 4)


def test_bare_tensor_output_records_no_index():
    m = BareBlock()
    _, cap = H.run_capture(m, lambda: m(torch.randn(2, 4)), names=["lin"])
    assert cap.data["lin"].took_index is None


def test_hooks_are_removed_even_when_the_forward_raises(stack):
    def boom():
        raise RuntimeError("forward exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        H.run_capture(stack, boom)
    # A leaked hook silently contaminates every later forward in the process.
    assert sum(len(m._forward_hooks) for m in stack.modules()) == 0


def test_a_missing_module_name_raises_rather_than_capturing_nothing(stack):
    """An empty capture looks exactly like a successful one that found nothing.
    Asking for a module that does not exist has to be loud."""
    with pytest.raises(KeyError, match="do not exist"):
        with H.capture(stack, names=["encoder_blocks.0", "nope.7"]):
            pass


def test_unknown_architecture_raises():
    class Mystery(nn.Module):
        pass
    with pytest.raises(KeyError, match="no residual read positions"):
        H.residual_points(Mystery())


def test_repeated_calls_keep_the_last_by_default(stack):
    """A module run once per diffusion step overwrites its entry unless asked
    otherwise; the count is still recorded so the overwrite is visible."""
    x = torch.randn(2, 4)
    with H.capture(stack, names=["encoder_blocks.0"]) as cap:
        with torch.no_grad():
            stack(x)
            stack(x)
    assert cap.call_counts["encoder_blocks.0"] == 2
    assert torch.is_tensor(cap["encoder_blocks.0"])


def test_keep_all_calls_accumulates(stack):
    x = torch.randn(2, 4)
    with H.capture(stack, names=["encoder_blocks.0"], keep_all_calls=True) as cap:
        with torch.no_grad():
            stack(x)
            stack(x)
    assert isinstance(cap["encoder_blocks.0"], list)
    assert len(cap["encoder_blocks.0"]) == 2


def test_captured_values_are_detached(stack):
    x = torch.randn(2, 4, requires_grad=True)
    _, cap = H.run_capture(stack, lambda: stack(x))
    for c in cap.data.values():
        assert not c.value.requires_grad


def test_layers_helper_returns_stack_order(stack):
    _, cap = H.run_capture(stack, lambda: stack(torch.randn(2, 4)))
    assert len(cap.layers(r"^decoder_blocks\.")) == 3


def test_first_tensor_rejects_a_tensorless_output():
    with pytest.raises(TypeError, match="no tensor"):
        H._first_tensor(("a", None, 3))
