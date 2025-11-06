
# i gotta create of have py file that will call all the features and there it would also load the data and then call all the yk features there and and for each of them and then for each class then we would have sth like that that would help us have the data for eahch of the classes i guess created and i think i gotta do that and extract sth like that for each of them happening so that is actually sth that i should create 
#i'm not quite srue if that should happen in .py or a jupyter notbook exactly so that is what i am gonna do 

from typing import Any, Dict, Iterable, Optional, Tuple, Union
import pandas as pd, numpy as np
from rna_features import RnaFeatures
from disease_features import DiseaseFeatures
from cross_features import CrossFeatures
try:
    from nn_features import NNFeatures  # optional
except Exception:
    NNFeatures = None  # ok if not present

MatrixLike = Union[np.ndarray, pd.DataFrame]

class FeatureExtractor:
    """Thin router over your feature modules."""

    _REGISTRY: Dict[str, Any] = {
        "rna": RnaFeatures,
        "disease": DiseaseFeatures,
        "cross": CrossFeatures,
    }
    if NNFeatures is not None:
        _REGISTRY["nn"] = NNFeatures

    @classmethod
    def list_domains(cls) -> str:
        return " | ".join(sorted(cls._REGISTRY.keys()))

    @classmethod
    def list_methods(cls, domain: str) -> str:
        mod = cls._module(domain)
        return mod.describe_methods()

    @classmethod
    def run(cls, domain: str, method_id: int, *args, **kwargs):
        """
        Delegates to <Module>.extract(method_id, *args, **kwargs).
        Keep args/kwargs flexible so you don't have to rename anything.
        """
        mod = cls._module(domain)
        return mod.extract(method_id, *args, **kwargs)

    @classmethod
    def _module(cls, domain: str):
        try:
            return cls._REGISTRY[domain.lower()]
        except KeyError:
            raise ValueError(f"Unknown domain '{domain}'. Known: {cls.list_domains()}")

def load_sequences_csv(path: str):
    df = pd.read_csv(path)
    if not {"id","seq"}.issubset(df.columns):
        raise ValueError("sequences.csv must have columns: id,seq")
    return df["id"].astype(str).tolist(), df["seq"].astype(str).tolist()
