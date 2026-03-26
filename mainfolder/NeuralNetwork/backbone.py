from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from typing import Any


@dataclass
class HFBackbone:
    model_id: str
    tokenizer: AutoTokenizer
    model: AutoModel
    hidden_size: int
    tokenizer_kwargs: dict[str, Any]
    max_input_bases: int | None = None
