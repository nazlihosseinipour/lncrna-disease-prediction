from transformers import AutoTokenizer, AutoModel
import torch
from backbone import HFBackbone

class BackboneRegistry:
    _cache = {}

    @classmethod
    def get(cls, model_id: str) -> HFBackbone:
        if model_id in cls._cache:
            return cls._cache[model_id]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            tmp = tokenizer("AUG", return_tensors="pt")
            with torch.no_grad():
                hs = model(**{k: v.to(device) for k, v in tmp.items()}).last_hidden_state
            hidden_size = int(hs.shape[-1])

        bb = HFBackbone(model_id, tokenizer, model, hidden_size)
        cls._cache[model_id] = bb
        return bb
