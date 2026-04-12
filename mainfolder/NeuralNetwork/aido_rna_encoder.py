from abc import ABC
from typing import Iterable, List

import pandas as pd
import torch

from mainfolder.utils.utils import to_rna, sliding_chunks
from mainfolder.utils.validators import (
    require_return_format,
    require_sample_ids_len,
    require_seqs,
)


class AIDORNAEncoder(ABC):
    """
    AIDO RNA encoder using the official ModelGenerator Embed task.

    This model is not packaged like a standard AutoTokenizer/AutoModel checkpoint,
    so it needs its own backend instead of the generic Hugging Face backbone path.
    """

    model_id = "genbio-ai/AIDO.RNA-1.6B"
    backbone_name = "aido_rna_1b600m"
    max_input_bases = 1024

    def __init__(self):
        try:
            from modelgenerator.tasks import Embed
        except ImportError as exc:
            raise ImportError(
                "AIDO requires the ModelGenerator package. "
                "Install the official AIDO/ModelGenerator inference dependencies "
                "before running AIDO NN methods."
            ) from exc

        self.model = Embed.from_config({"model.backbone": self.backbone_name}).eval()
        self._hidden_size = None
        self._logged_length_events = set()

    def normalize(self, seqs: Iterable[str]) -> List[str]:
        return [to_rna(s) for s in seqs]

    def _log_length_event(self, original_len: int, truncated_len: int, skipped: bool) -> None:
        key = (original_len, truncated_len, skipped)
        if key in self._logged_length_events:
            return
        self._logged_length_events.add(key)
        print(
            f"[nn] model={self.model_id} original_len={original_len} "
            f"truncated_len={truncated_len} skipped={skipped}"
        )

    def _prepare_batch_sequences(self, batch: List[str]) -> List[str]:
        prepared = []
        for seq in batch:
            original_len = len(seq)
            if original_len > self.max_input_bases:
                prepared.append(seq[: self.max_input_bases])
                self._log_length_event(original_len, self.max_input_bases, False)
            else:
                prepared.append(seq)
        return prepared

    @staticmethod
    def _batch(items: List[str], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _embed_batch(self, batch: List[str]) -> torch.Tensor:
        safe_batch = self._prepare_batch_sequences(batch)

        if hasattr(self.model, "transform"):
            collated = self.model.transform({"sequences": safe_batch})
        elif hasattr(self.model, "collate"):
            collated = self.model.collate({"sequences": safe_batch})
        else:
            raise AttributeError("AIDO Embed model does not expose transform/collate.")

        with torch.no_grad():
            out = self.model(collated)

        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "detach"):
            out = out.detach()
        elif hasattr(out, "logits"):
            out = out.logits.detach()
        else:
            raise TypeError(f"Unexpected AIDO output type: {type(out)!r}")

        if out.ndim != 2:
            raise ValueError(
                f"AIDO Embed is expected to return [batch, hidden] embeddings, got shape {tuple(out.shape)}"
            )

        self._hidden_size = int(out.shape[-1])
        return out.cpu()

    @torch.no_grad()
    def sequence_embeddings(self, sequences, return_format="matrix", sample_ids=None, batch_size=8):
        require_seqs(sequences)
        require_return_format(return_format)
        seqs = self.normalize(sequences)
        rows = []
        for batch in self._batch(seqs, batch_size):
            rows.extend(self._embed_batch(batch).tolist())
        hidden = self._hidden_size or 0
        labels = [f"f{i}" for i in range(hidden)]
        if return_format == "matrix":
            return labels, rows
        df = pd.DataFrame(rows, columns=labels)
        if sample_ids is not None:
            sample_ids = list(sample_ids)
            require_sample_ids_len(sample_ids, len(seqs))
            df.insert(0, "sample_id", sample_ids)
        return labels, df

    @torch.no_grad()
    def token_embeddings(self, sequences, layer=None, return_format="matrix", sample_ids=None, batch_size=8):
        raise NotImplementedError(
            "AIDO token embeddings are not exposed by the current ModelGenerator Embed backend."
        )

    @torch.no_grad()
    def sequence_embeddings_chunked(
        self,
        sequences,
        window=1024,
        stride=512,
        agg="mean",
        return_format="matrix",
        sample_ids=None,
        batch_size=8,
    ):
        require_seqs(sequences)
        require_return_format(return_format)
        seqs = self.normalize(sequences)
        if window > self.max_input_bases:
            self._log_length_event(window, self.max_input_bases, False)
            window = self.max_input_bases
            stride = min(stride, window)
        rows = []
        for seq in seqs:
            pieces = sliding_chunks(seq, window, stride)
            piece_rows = []
            for batch in self._batch(pieces, batch_size):
                piece_rows.append(self._embed_batch(batch))
            if not piece_rows:
                hidden = self._hidden_size or 0
                rows.append([0.0] * hidden)
                continue
            mat = torch.cat(piece_rows, dim=0)
            vec = mat.mean(0) if agg == "mean" else mat.max(0).values
            rows.append(vec.tolist())
        hidden = self._hidden_size or 0
        labels = [f"f{i}" for i in range(hidden)]
        if return_format == "matrix":
            return labels, rows
        df = pd.DataFrame(rows, columns=labels)
        if sample_ids is not None:
            sample_ids = list(sample_ids)
            require_sample_ids_len(sample_ids, len(seqs))
            df.insert(0, "sample_id", sample_ids)
        return labels, df
