import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mainfolder.NeuralNetwork.abstract_rna_encoder import AbstractRNAEncoder
from mainfolder.NeuralNetwork.aido_rna_encoder import AIDORNAEncoder
from mainfolder.NeuralNetwork.backbone import HFBackbone
from mainfolder.NeuralNetwork.backbone_registry import BackboneRegistry
import mainfolder.NeuralNetwork.backbone_registry as backbone_registry_module


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, batch, **kwargs):
        self.calls.append((list(batch), dict(kwargs)))
        max_length = kwargs.get("max_length") or max(len(x) for x in batch)
        batch_size = len(batch)
        return {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
        }


class DummyModel:
    def __init__(self, hidden_size=4):
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def to(self, device):
        self._param = torch.nn.Parameter(self._param.to(device))
        return self

    def eval(self):
        return self

    def parameters(self):
        yield self._param

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones((batch_size, seq_len, self.config.hidden_size), dtype=torch.float32, device=input_ids.device)
        output = type("Output", (), {"last_hidden_state": hidden})()
        if output_hidden_states:
            output.hidden_states = [hidden, hidden]
        return output


class DummyMPRNAEncoder(AbstractRNAEncoder):
    model_id = "yangheng/MP-RNA"


def test_mp_rna_sequences_are_clipped_before_tokenization(monkeypatch, capsys):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    backbone = HFBackbone(
        model_id="yangheng/MP-RNA",
        tokenizer=tokenizer,
        model=model,
        hidden_size=4,
        tokenizer_kwargs={
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 512,
            "padding": "max_length",
        },
        max_input_bases=512,
    )
    monkeypatch.setattr(BackboneRegistry, "get", classmethod(lambda cls, model_id: backbone))

    encoder = DummyMPRNAEncoder()
    labels, rows = encoder.sequence_embeddings(["A" * 900], batch_size=1)

    assert len(labels) == 4
    assert len(rows) == 1
    batch, kwargs = tokenizer.calls[0]
    assert len(batch[0]) == 512
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 512
    assert kwargs["padding"] == "max_length"
    assert kwargs["return_tensors"] == "pt"
    captured = capsys.readouterr()
    assert "original_len=900" in captured.out
    assert "truncated_len=512" in captured.out
    assert "skipped=False" in captured.out


def test_backbone_registry_uses_trust_remote_code(monkeypatch):
    calls = {}

    def fake_tokenizer_from_pretrained(model_id, **kwargs):
        calls["tokenizer"] = {"model_id": model_id, **kwargs}
        return DummyTokenizer()

    def fake_model_from_pretrained(model_id, **kwargs):
        calls["model"] = {"model_id": model_id, **kwargs}
        return DummyModel()

    monkeypatch.setattr(backbone_registry_module.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(backbone_registry_module.AutoModel, "from_pretrained", fake_model_from_pretrained)
    BackboneRegistry._cache.clear()

    bb = BackboneRegistry.get("yangheng/MP-RNA")

    assert bb.max_input_bases == 512
    assert bb.tokenizer_kwargs["max_length"] == 512
    assert calls["tokenizer"]["trust_remote_code"] is True
    assert calls["tokenizer"]["use_fast"] is False
    assert calls["model"]["trust_remote_code"] is True


def test_transformers_compat_shim_installs_missing_prune_helper(monkeypatch):
    import transformers.pytorch_utils as pytorch_utils

    had_attr = hasattr(pytorch_utils, "find_pruneable_heads_and_indices")
    original = getattr(pytorch_utils, "find_pruneable_heads_and_indices", None)
    if had_attr:
        monkeypatch.delattr(pytorch_utils, "find_pruneable_heads_and_indices", raising=False)

    backbone_registry_module._install_transformers_compat_shims()

    assert hasattr(pytorch_utils, "find_pruneable_heads_and_indices")
    heads, index = pytorch_utils.find_pruneable_heads_and_indices({0, 2}, 4, 8, set())
    assert heads == {0, 2}
    assert index.numel() == 16

    if had_attr and original is not None:
        monkeypatch.setattr(pytorch_utils, "find_pruneable_heads_and_indices", original, raising=False)


def install_fake_modelgenerator(monkeypatch):
    calls = {"config": None, "batches": []}

    class FakeEmbedModel:
        def eval(self):
            return self

        def transform(self, batch):
            calls["batches"].append(list(batch["sequences"]))
            return batch

        def __call__(self, batch):
            n = len(batch["sequences"])
            return torch.ones((n, 4), dtype=torch.float32)

    class FakeEmbed:
        @classmethod
        def from_config(cls, config):
            calls["config"] = dict(config)
            return FakeEmbedModel()

    modelgenerator_mod = types.ModuleType("modelgenerator")
    tasks_mod = types.ModuleType("modelgenerator.tasks")
    tasks_mod.Embed = FakeEmbed
    modelgenerator_mod.tasks = tasks_mod
    monkeypatch.setitem(sys.modules, "modelgenerator", modelgenerator_mod)
    monkeypatch.setitem(sys.modules, "modelgenerator.tasks", tasks_mod)
    return calls


def test_aido_sequences_are_clipped_before_embedding(monkeypatch, capsys):
    calls = install_fake_modelgenerator(monkeypatch)

    encoder = AIDORNAEncoder()
    labels, rows = encoder.sequence_embeddings(["A" * 1800], batch_size=1)

    assert calls["config"] == {"model.backbone": "aido_rna_1b600m"}
    assert len(labels) == 4
    assert len(rows) == 1
    assert len(calls["batches"][0][0]) == 1024
    captured = capsys.readouterr()
    assert "original_len=1800" in captured.out
    assert "truncated_len=1024" in captured.out
    assert "skipped=False" in captured.out


def test_aido_token_embeddings_not_supported_yet(monkeypatch):
    install_fake_modelgenerator(monkeypatch)
    encoder = AIDORNAEncoder()

    try:
        encoder.token_embeddings(["AUGC"])
    except NotImplementedError as exc:
        assert "not exposed" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
