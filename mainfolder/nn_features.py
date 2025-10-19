"""
Facade that exposes numeric method IDs and delegates to concrete encoders.
Works with:
  - mp_rna_encoder.MPRNAEncoder
  - aido_rna_encoder.AIDORNAEncoder
Requires:
  - abstract_rna_encoder.py (shared logic lives there)
  - backbone.py / backbone_registry.py (HF loader/cache)

"""
from feature_module import FeatureModule         #parentclass
from nn.mp_rna_encoder import MPRNAEncoder
from nn.aido_rna_encoder import AIDORNAEncoder



# Singletons so the heavy HF models are loaded once per process
_MP = MPRNAEncoder()
_AIDO = AIDORNAEncoder()


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
        return _MP.sequence_embeddings(
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
        return _MP.token_embeddings(
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
        return _MP.sequence_embeddings_chunked(
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
        return _AIDO.sequence_embeddings(
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
        return _AIDO.token_embeddings(
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
        return _AIDO.sequence_embeddings_chunked(
            sequences=sequences,
            window=window,
            stride=stride,
            agg=agg,
            return_format=return_format,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )
