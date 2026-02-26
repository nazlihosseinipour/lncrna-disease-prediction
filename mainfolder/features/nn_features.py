"""
Facade that exposes numeric method IDs and delegates to concrete encoders.
Works with:
  - MPRNAEncoder (yangheng/MP-RNA)
  - AIDORNAEncoder (genbio-ai/AIDO.RNA-1.6B)
Backbone logic lives in mainfolder.nn (abstract_rna_encoder, backbone, registry).
"""
from mainfolder.core.feature_module import FeatureModule  # parent class
from mainfolder.NeuralNetwork.mp_rna_encoder import MPRNAEncoder
from mainfolder.NeuralNetwork.aido_rna_encoder import AIDORNAEncoder

# Lazy singletons so we don't download heavy HF models at import time
_MP = None
_AIDO = None


def _mp():
    global _MP
    if _MP is None:
        _MP = MPRNAEncoder()
    return _MP


def _aido():
    global _AIDO
    if _AIDO is None:
        _AIDO = AIDORNAEncoder()
    return _AIDO


class NNFeatures(FeatureModule):
    """
    RNA neural-network features facade.

    Each method returns:
      - (labels: List[str], matrix: List[List[float]]) when return_format="matrix"
      - (labels: List[str], df: pandas.DataFrame) when return_format="dataframe"
    See abstract_rna_encoder.py for the exact shapes and behavior.
    """

    # Keep IDs stable across your codebase
    METHOD_MAP = {
        # MP-RNA (yangheng/MP-RNA)
        100: "mp_sequence",            # one vector per sequence (mean-pooled)
        101: "mp_tokens",              # per-nucleotide embeddings
        130: "mp_sequence_chunked",    # windowed agg for long sequences

        # AIDO.RNA-1.6B (genbio-ai/AIDO.RNA-1.6B)
        103: "aido_sequence",
        104: "aido_tokens",
        131: "aido_sequence_chunked",
    }

    @classmethod
    def extract(cls, method_id, *args, **kwargs):
        # Defer to base FeatureModule’s reflection-based dispatcher.
        # It will map method_id -> METHOD_MAP[method_id] -> method on this class.
        return super().extract(method_id, *args, **kwargs)

    #  MP-RNA delegates 

    @staticmethod
    def mp_sequence(
        sequences,
        return_format="matrix",
        sample_ids=None,
        batch_size=16,
        **kwargs,
    ):
        """MP-RNA, sequence-level embeddings (mean pooled)."""
        return _mp().sequence_embeddings(
            sequences=sequences,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )

    @staticmethod
    def mp_tokens(
        sequences,
        layer=None,
        return_format="matrix",
        sample_ids=None,
        batch_size=8,
        **kwargs,
    ):
        """MP-RNA, per-token embeddings (optionally choose a hidden layer)."""
        return _mp().token_embeddings(
            sequences=sequences,
            layer=layer,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )

    @staticmethod
    def mp_sequence_chunked(
        sequences,
        window=1024,
        stride=512,
        agg="mean",
        return_format="matrix",
        sample_ids=None,
        batch_size=8,
        **kwargs,
    ):
        """MP-RNA, sliding window (long sequences) with mean/max aggregation."""
        return _mp().sequence_embeddings_chunked(
            sequences=sequences,
            window=window,
            stride=stride,
            agg=agg,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )

    # AIDO.RNA-1.6B delegates

    @staticmethod
    def aido_sequence(
        sequences,
        return_format="matrix",
        sample_ids=None,
        batch_size=8,
        **kwargs,
    ):
        """AIDO.RNA-1.6B, sequence-level embeddings (mean pooled)."""
        return _aido().sequence_embeddings(
            sequences=sequences,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )

    @staticmethod
    def aido_tokens(
        sequences,
        layer=None,
        return_format="matrix",
        sample_ids=None,
        batch_size=4,
        **kwargs,
    ):
        """AIDO.RNA-1.6B, per-token embeddings (optionally choose a hidden layer)."""
        return _aido().token_embeddings(
            sequences=sequences,
            layer=layer,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )

    @staticmethod
    def aido_sequence_chunked(
        sequences,
        window=1024,
        stride=512,
        agg="mean",
        return_format="matrix",
        sample_ids=None,
        batch_size=4,
        **kwargs,
    ):
        """AIDO.RNA-1.6B, sliding window (long sequences) with mean/max aggregation."""
        return _aido().sequence_embeddings_chunked(
            sequences=sequences,
            window=window,
            stride=stride,
            agg=agg,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )
