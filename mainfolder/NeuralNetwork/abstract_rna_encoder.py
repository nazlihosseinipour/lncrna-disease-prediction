from typing import Iterable, List, Optional, Literal
from abc import ABC
import pandas as pd
import torch
from .backbone_registry import BackboneRegistry
from mainfolder.utils.validators import require_seqs, require_return_format, require_sample_ids_len
from mainfolder.utils.utils import to_rna, mean_pool, sliding_chunks

class AbstractRNAEncoder(ABC): 
        model_id: str  # subclass sets this

        def __init__(self):
            self.backbone = BackboneRegistry.get(self.model_id)
            self.device = next(self.backbone.model.parameters()).device
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
            max_len = self.backbone.max_input_bases
            if max_len is None:
                return batch

            prepared = []
            for seq in batch:
                original_len = len(seq)
                if original_len > max_len:
                    prepared.append(seq[:max_len])
                    self._log_length_event(original_len, max_len, False)
                else:
                    prepared.append(seq)
            return prepared

        def _tokenize_batch(self, batch: List[str]):
            safe_batch = self._prepare_batch_sequences(batch)
            enc = self.backbone.tokenizer(safe_batch, **self.backbone.tokenizer_kwargs)
            return {k: v.to(self.device) for k, v in enc.items()}

        @staticmethod
        def _batch(items: List[str], size: int):
            for i in range(0, len(items), size):
                yield items[i:i+size]

        @torch.no_grad()
        def sequence_embeddings(self, sequences, return_format="matrix", sample_ids=None, batch_size=16):
            require_seqs(sequences); require_return_format(return_format)
            seqs = self.normalize(sequences); bb = self.backbone
            rows = []
            for batch in self._batch(seqs, batch_size):
                enc = self._tokenize_batch(batch)
                out = bb.model(**enc)
                rows.extend(mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().tolist())
            labels = [f"f{i}" for i in range(bb.hidden_size)]
            if return_format == "matrix": return labels, rows
            df = pd.DataFrame(rows, columns=labels)
            if sample_ids is not None:
                sample_ids = list(sample_ids); require_sample_ids_len(sample_ids, len(seqs))
                df.insert(0, "sample_id", sample_ids)
            return labels, df

        @torch.no_grad()
        def token_embeddings(self, sequences, layer=None, return_format="matrix", sample_ids=None, batch_size=8):
            require_seqs(sequences); require_return_format(return_format)
            seqs = self.normalize(sequences); bb = self.backbone
            labels = [f"f{i}" for i in range(bb.hidden_size)]
            if return_format == "matrix":
                ragged = []
                for batch in self._batch(seqs, batch_size):
                    enc = self._tokenize_batch(batch)
                    out = bb.model(**enc, output_hidden_states=(layer is not None))
                    tok = out.hidden_states[layer] if layer is not None else out.last_hidden_state
                    for i in range(tok.shape[0]):
                        ragged.append(tok[i].cpu().reshape(-1).tolist())
                return labels, ragged
            # long-form DF
            rows = []
            sids = list(sample_ids) if sample_ids is not None else [f"s{i}" for i in range(len(seqs))]
            if sample_ids is not None: require_sample_ids_len(sample_ids, len(seqs))
            for i0 in range(0, len(seqs), batch_size):
                batch = seqs[i0:i0+batch_size]
                enc = self._tokenize_batch(batch)
                out = bb.model(**enc, output_hidden_states=(layer is not None))
                tok = out.hidden_states[layer] if layer is not None else out.last_hidden_state
                attn = enc["attention_mask"].cpu(); tok = tok.cpu()
                for i in range(tok.shape[0]):
                    L = int(attn[i].sum().item()); sid = sids[i0 + i]
                    for t in range(L):
                        rows.append({"sample_id": sid, "token_index": t, **{f"f{j}": float(tok[i, t, j]) for j in range(tok.shape[-1])}})
            return labels, pd.DataFrame.from_records(rows)

        @torch.no_grad()
        def sequence_embeddings_chunked(self, sequences, window=1024, stride=512, agg="mean",
                                        return_format="matrix", sample_ids=None, batch_size=8):
            require_seqs(sequences); require_return_format(return_format)
            seqs = self.normalize(sequences); bb = self.backbone
            if bb.max_input_bases is not None and window > bb.max_input_bases:
                self._log_length_event(window, bb.max_input_bases, False)
                window = bb.max_input_bases
                stride = min(stride, window)
            rows = []
            for s in seqs:
                pieces = sliding_chunks(s, window, stride)
                emb = []
                for batch in self._batch(pieces, batch_size):
                    enc = self._tokenize_batch(batch)
                    out = bb.model(**enc)
                    emb.append(mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu())
                if not emb:
                    rows.append([0.0]*bb.hidden_size); continue
                mat = torch.cat(emb, dim=0)
                vec = mat.mean(0) if agg == "mean" else mat.max(0).values
                rows.append(vec.tolist())
            labels = [f"f{i}" for i in range(bb.hidden_size)]
            if return_format == "matrix": return labels, rows
            df = pd.DataFrame(rows, columns=labels)
            if sample_ids is not None:
                sample_ids = list(sample_ids); require_sample_ids_len(sample_ids, len(seqs))
                df.insert(0, "sample_id", sample_ids)
            return labels, df
