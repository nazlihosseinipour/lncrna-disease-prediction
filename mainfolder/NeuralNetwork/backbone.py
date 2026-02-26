from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel


@dataclass
class HFBackbone:
    model_id: str
    tokenizer: AutoTokenizer
    model: AutoModel
    hidden_size: int