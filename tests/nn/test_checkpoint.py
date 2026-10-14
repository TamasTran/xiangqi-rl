from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from xiangqi_rl.nn import (  # noqa: E402
    ModelConfig,
    create_model,
    load_checkpoint,
    load_model,
    save_checkpoint,
)

CFG = ModelConfig(hidden_channels=16, num_res_blocks=1, norm_groups=4, policy_channels=4,
                  value_channels=4, value_hidden=16, norm="batch")


def test_roundtrip_preserves_config_and_outputs(tmp_path) -> None:
    model = create_model(CFG, seed=0).eval()
    for p in model.parameters():  # make weights non-trivial
        p.data.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(3, 17, 10, 9)
    path = tmp_path / "sub" / "model.pt"
    save_checkpoint(path, model, extra={"step": 12})
    loaded = load_model(path).eval()
    assert loaded.config == CFG
    a, b = model(x), loaded(x)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)
    assert load_checkpoint(path)["extra"] == {"step": 12}
    assert not list(path.parent.glob("*.tmp"))


def test_batchnorm_buffers_are_saved(tmp_path) -> None:
    model = create_model(CFG, seed=0).train()
    model(torch.randn(8, 17, 10, 9) + 3)  # updates running statistics
    save_checkpoint(tmp_path / "m.pt", model)
    loaded = load_model(tmp_path / "m.pt")
    for (n1, b1), (_, b2) in zip(model.named_buffers(), loaded.named_buffers()):
        assert torch.equal(b1, b2), n1


def test_optimizer_state_is_optional(tmp_path) -> None:
    model = create_model(CFG, seed=0)
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 17, 10, 9))[0].sum().backward()
    opt.step()
    save_checkpoint(tmp_path / "with.pt", model, optimizer=opt)
    save_checkpoint(tmp_path / "without.pt", model)
    assert "optimizer_state_dict" in load_checkpoint(tmp_path / "with.pt")
    assert "optimizer_state_dict" not in load_checkpoint(tmp_path / "without.pt")
    fresh = torch.optim.Adam(create_model(CFG, seed=1).parameters())
    fresh.load_state_dict(load_checkpoint(tmp_path / "with.pt")["optimizer_state_dict"])


def test_bad_checkpoints_rejected(tmp_path) -> None:
    torch.save({"hello": 1}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="not a XiangqiRL"):
        load_checkpoint(tmp_path / "x.pt")
    model = create_model(CFG, seed=0)
    save_checkpoint(tmp_path / "ok.pt", model)
    payload = load_checkpoint(tmp_path / "ok.pt")
    payload["architecture"] = "other"
    torch.save(payload, tmp_path / "arch.pt")
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(tmp_path / "arch.pt")
    payload["architecture"] = "xiangqi-policy-value-resnet-v1"
    payload["format_version"] = 99
    torch.save(payload, tmp_path / "ver.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(tmp_path / "ver.pt")


def test_untrusted_pickle_is_refused(tmp_path) -> None:
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    path = tmp_path / "evil.pt"
    with open(path, "wb") as fh:
        pickle.dump({"model_state_dict": Evil()}, fh)
    with pytest.raises(Exception):
        load_checkpoint(path)


def test_cuda_checkpoint_loads_on_cpu(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = create_model(CFG, seed=0).to("cuda")
    save_checkpoint(tmp_path / "cuda.pt", model)
    loaded = load_model(tmp_path / "cuda.pt", map_location="cpu")
    assert next(loaded.parameters()).device.type == "cpu"
