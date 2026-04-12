import types

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from .backbone import HFBackbone


def _compat_find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    Compatibility shim for remote model code that still imports
    find_pruneable_heads_and_indices from transformers.pytorch_utils.
    """
    heads = set(heads) - set(already_pruned_heads)
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(mask.numel(), device=mask.device)[mask].long()
    return heads, index


def _install_transformers_compat_shims() -> None:
    try:
        import transformers.pytorch_utils as pytorch_utils

        if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
            pytorch_utils.find_pruneable_heads_and_indices = _compat_find_pruneable_heads_and_indices
    except Exception:
        pass

    try:
        import transformers.modeling_utils as modeling_utils

        if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
            modeling_utils.find_pruneable_heads_and_indices = _compat_find_pruneable_heads_and_indices
    except Exception:
        pass


def _patch_remote_config(config):
    """
    Fill newer transformers config attributes that older remote checkpoints may
    omit, but current modeling code expects to exist.
    """
    defaults = {
        "is_decoder": False,
        "add_cross_attention": False,
        "chunk_size_feed_forward": 0,
        "tie_word_embeddings": True,
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    return config


def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is None:
        return [None] * num_hidden_layers

    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise AssertionError(f"head_mask.dim != 5, got {head_mask.dim()}")

    try:
        dtype = next(self.parameters()).dtype
    except Exception:
        dtype = torch.float32
    head_mask = head_mask.to(dtype=dtype)
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask


def _patch_remote_model_instance(model):
    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = types.MethodType(_compat_get_head_mask, model)
    return model


class BackboneRegistry:
    _cache = {}
    _TOKENIZER_CONFIGS = {
        "yangheng/MP-RNA": {
            "tokenizer_kwargs": {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": 512,
                "padding": "max_length",
            },
            "max_input_bases": 512,
        },
        "genbio-ai/AIDO.RNA-1.6B": {
            "tokenizer_kwargs": {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": 1024,
                "padding": "max_length",
            },
            "max_input_bases": 1024,
        },
    }

    @classmethod
    def get(cls, model_id: str) -> HFBackbone:
        if model_id in cls._cache:
            return cls._cache[model_id]

        _install_transformers_compat_shims()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        cfg = cls._TOKENIZER_CONFIGS.get(
            model_id,
            {"tokenizer_kwargs": {"return_tensors": "pt", "padding": True, "truncation": True}, "max_input_bases": None},
        )
        config = _patch_remote_config(
            AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            model_id,
            config=config,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        model = _patch_remote_model_instance(model)

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            tmp = tokenizer("AUG", return_tensors="pt", truncation=True, max_length=cfg["max_input_bases"] or 512)
            with torch.no_grad():
                hs = model(**{k: v.to(device) for k, v in tmp.items()}).last_hidden_state
            hidden_size = int(hs.shape[-1])

        bb = HFBackbone(
            model_id,
            tokenizer,
            model,
            hidden_size,
            tokenizer_kwargs=dict(cfg["tokenizer_kwargs"]),
            max_input_bases=cfg["max_input_bases"],
        )
        cls._cache[model_id] = bb
        return bb
