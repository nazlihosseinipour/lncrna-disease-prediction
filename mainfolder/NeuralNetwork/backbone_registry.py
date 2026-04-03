from transformers import AutoTokenizer, AutoModel
import torch
from .backbone import HFBackbone

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        cfg = cls._TOKENIZER_CONFIGS.get(
            model_id,
            {"tokenizer_kwargs": {"return_tensors": "pt", "padding": True, "truncation": True}, "max_input_bases": None},
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device).eval()

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
