import argparse
from pathlib import Path
import pandas as pd
import sys
from typing import Dict
from mainfolder.utils.loader import (
    load_sequences,
    load_txt_list,
    load_csv_df,
    load_edges_child_parent,
    load_sequences_csv,
    preprocess_sequences,
    save_output,
)
from mainfolder.core.feature_extractor import FeatureExtractor
from mainfolder.features.rna_features import RnaFeatures
from mainfolder.utils.utils import ALPHABET


class CLI:
    def __init__(self):
        p = argparse.ArgumentParser(description="Feature extraction CLI")
        sp = p.add_subparsers(dest="domain", required=True)

        #RNA
        rna = sp.add_parser("rna", help="RNA feature extraction")
        rna.add_argument("--list", action="store_true", help="List available methods and exit")
        rna.add_argument("--method", type=int, default=1, help="Method ID to run")

        # mutually exclusive: TXT vs CSV
        src = rna.add_mutually_exclusive_group(required=True)
        src.add_argument("--seqs", help="TXT: one sequence per line")
        src.add_argument("--seqs_csv", help="CSV: columns id,seq")

        rna.add_argument("--sample_ids", help="TXT: one ID per line (only with --seqs)")
        rna.add_argument("--k", type=int, help="k-mer size (methods 1,2,8,9)")
        rna.add_argument("--lam", type=int, help="lag parameter for PseDNC (method 3)")
        rna.add_argument("--weight", type=float, help="weight parameter for PseDNC (method 3)")
        rna.add_argument("--L", type=int, help="lag parameter for DAC/DCC/DACC (methods 4,5,6)")
        rna.add_argument("--k_gap", type=int, help="gap parameter for monoMonoKGap/monoDiKGap (methods 11,12)")
        rna.add_argument("--props_csv", help="CSV with dinucleotide properties (col 'dinuc', rest = features)")
        rna.add_argument("--normalize", type=lambda s: s.lower() != "false", default=True)
        rna.add_argument("--return_format", choices=["matrix", "dataframe"], default="dataframe")

        # optional preprocessing flags
        rna.add_argument("--no_upper", action="store_true", help="Do not uppercase sequences")
        rna.add_argument("--keep_t", action="store_true", help="Do not convert T->U")
        rna.add_argument("--strict", action="store_true", help="Error on invalid chars instead of dropping")
        rna.add_argument("--save_clean", help="Optional path to save cleaned sequences as CSV (id,seq)")

        rna.add_argument("-o", "--output", help="Path to save output")

        #RNA-ALL
        rna_all = sp.add_parser("rna-all", help="Run all RNA feature methods")
        src_all = rna_all.add_mutually_exclusive_group(required=True)
        src_all.add_argument("--seqs", help="TXT: one sequence per line")
        src_all.add_argument("--seqs_csv", help="CSV: columns id,seq")
        rna_all.add_argument("--sample_ids", help="TXT: one ID per line (only with --seqs)")
        rna_all.add_argument("--k", type=int, default=3, help="k-mer size for methods 1/2")
        rna_all.add_argument("--lam", type=int, default=2, help="lam for PseDNC (method 3)")
        rna_all.add_argument("--weight", type=float, default=0.5, help="weight for PseDNC (method 3)")
        rna_all.add_argument("--L", type=int, default=3, help="lag for DAC/DCC/DACC (methods 4,5,6)")
        rna_all.add_argument("--k_gap", type=int, default=1, help="gap for monoMonoKGap/monoDiKGap (methods 11,12)")
        rna_all.add_argument("--props_csv", help="CSV with dinucleotide properties (col 'dinuc', rest = features)")
        rna_all.add_argument("--normalize", type=lambda s: s.lower() != "false", default=True)
        rna_all.add_argument("--return_format", choices=["matrix", "dataframe"], default="dataframe")
        rna_all.add_argument("--no_upper", action="store_true", help="Do not uppercase sequences")
        rna_all.add_argument("--keep_t", action="store_true", help="Do not convert T->U")
        rna_all.add_argument("--strict", action="store_true", help="Error on invalid chars instead of dropping")
        rna_all.add_argument("--outdir", default="final-output", help="Output directory for all RNA feature CSVs")

        #Disease
        dis = sp.add_parser("disease", help="Disease feature extraction")
        dis.add_argument("--list", action="store_true")
        dis.add_argument("--method", type=int, default=13)
        dis.add_argument("--edges")
        dis.add_argument("--edge_weight", type=float, default=0.8)
        dis.add_argument("--term_a")
        dis.add_argument("--term_b")
        dis.add_argument("--disease_terms")
        dis.add_argument("--Y")
        dis.add_argument("--disease_sim")
        dis.add_argument("-o", "--output")

        #Cross
        cr = sp.add_parser("cross", help="Cross features (GIP/SVD)")
        cr.add_argument("--list", action="store_true")
        cr.add_argument("--method", type=int, choices=[16, 17], required=True)
        cr.add_argument("--matrix")
        cr.add_argument("--k", type=int, default=64)
        cr.add_argument("-o", "--output")

        #NN
        nn = sp.add_parser("nn", help="Neural network RNA feature extraction")
        nn.add_argument("--list", action="store_true", help="List available methods and exit")
        nn.add_argument("--method", type=int, default=100, help="Method ID to run")
        src_nn = nn.add_mutually_exclusive_group(required=True)
        src_nn.add_argument("--seqs", help="TXT: one sequence per line")
        src_nn.add_argument("--seqs_csv", help="CSV: columns id,seq")
        nn.add_argument("--sample_ids", help="TXT: one ID per line (only with --seqs)")
        nn.add_argument("--return_format", choices=["matrix", "dataframe"], default="dataframe")
        nn.add_argument("--batch_size", type=int, default=8)
        nn.add_argument("--layer", type=int, help="Hidden layer index for *_tokens methods")
        nn.add_argument("--window", type=int, default=1024, help="Window size for chunked methods")
        nn.add_argument("--stride", type=int, default=512, help="Stride for chunked methods")
        nn.add_argument("--agg", choices=["mean", "max"], default="mean", help="Aggregation for chunked methods")
        nn.add_argument("--no_upper", action="store_true", help="Do not uppercase sequences")
        nn.add_argument("--keep_t", action="store_true", help="Do not convert T->U")
        nn.add_argument("--strict", action="store_true", help="Error on invalid chars instead of dropping")
        nn.add_argument("--save_clean", help="Optional path to save cleaned sequences as CSV (id,seq)")
        nn.add_argument("-o", "--output", help="Path to save output")

        self.parser = p

    @staticmethod
    def _load_props_csv(path: str):
        """
        Load dinucleotide properties from CSV with columns:
        dinuc,p1,p2,...
        """
        import pandas as pd

        df = pd.read_csv(path)
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        if "dinuc" not in cols:
            raise ValueError("props_csv must include a 'dinuc' column")
        df["dinuc"] = df["dinuc"].str.upper()
        props = {}
        for _, row in df.iterrows():
            dinuc = row["dinuc"]
            vals = [float(x) for x in row.drop(labels="dinuc").tolist()]
            props[dinuc] = vals
        if not props:
            raise ValueError("props_csv contained no rows")
        return props


    def run(self, argv=None):
        args = self.parser.parse_args(argv)

        # List available methods
        if getattr(args, "list", False):
            print(FeatureExtractor.list_methods(args.domain))
            return

        #RNA class
        if args.domain == "rna":
            props = self._load_props_csv(args.props_csv) if args.props_csv else None

            # Load sequences
            if args.seqs_csv:
                ids, seqs = load_sequences_csv(args.seqs_csv)
            else:
                seqs = load_sequences(args.seqs)
                ids = load_txt_list(args.sample_ids)
                if ids is not None and len(ids) != len(seqs):
                    raise ValueError(
                        f"sample_ids length ({len(ids)}) != sequences length ({len(seqs)})"
                    )

            # Preprocess
            ids2, seqs2 = preprocess_sequences(
                ids,
                seqs,
                to_upper=not args.no_upper,
                replace_t_with_u=not args.keep_t,
                strict=args.strict,
            )

            # Save cleaned sequences (optional)
            if args.save_clean:
                if ids2 is None:
                    ids2 = [f"s{i}" for i in range(len(seqs2))]
                pd.DataFrame({"id": ids2, "seq": seqs2}).to_csv(args.save_clean, index=False)
                print(f"[saved] cleaned sequences -> {args.save_clean}", file=sys.stderr)

            # Run feature extraction
            kwargs = {
                "normalize": args.normalize,
                "return_format": args.return_format,
                "sample_ids": ids2,
            }
            if args.k is not None:
                kwargs["k"] = args.k
            if args.lam is not None:
                kwargs["lam"] = args.lam
            if args.weight is not None:
                kwargs["w"] = args.weight
            if getattr(args, "L", None) is not None:
                kwargs["L"] = args.L
            if args.k_gap is not None:
                kwargs["k_gap"] = args.k_gap
            if props is not None:
                kwargs["props"] = props

            obj = FeatureExtractor.run("rna", args.method, seqs2, **kwargs)
            save_output(obj, args.output)

        #Disease class
        elif args.domain == "disease":
            m = args.method
            if m == 13:
                edges = load_edges_child_parent(args.edges)
                obj = FeatureExtractor.run(
                    "disease",
                    13,
                    args.term_a,
                    args.term_b,
                    edges_child_parent=edges,
                    edge_weight=args.edge_weight,
                )
            elif m == 14:
                edges = load_edges_child_parent(args.edges)
                dtt = pd.read_csv(args.disease_terms)
                dtt.columns = [c.lower() for c in dtt.columns]
                disease_to_terms: Dict[str, list] = {
                    d: [t.strip() for t in g["term"].astype(str).tolist() if t.strip()]
                    for d, g in dtt.groupby("disease")
                }
                obj = FeatureExtractor.run(
                    "disease",
                    14,
                    disease_to_terms=disease_to_terms,
                    edges_child_parent=edges,
                    edge_weight=args.edge_weight,
                )
            elif m == 15:
                Y = load_csv_df(args.Y)
                D = load_csv_df(args.disease_sim)
                obj = FeatureExtractor.run("disease", 15, Y=Y, disease_sim=D)
            else:
                raise ValueError(f"Unsupported disease method {m}")
            save_output(obj, args.output)

        #Cross class
        elif args.domain == "cross":
            M = load_csv_df(args.matrix)
            # If matrix has an ID column, drop it to keep numeric values only
            for id_col in ("ID", "id"):
                if id_col in M.columns:
                    M = M.set_index(id_col, drop=True)
                    break
            # Keep only numeric columns (drop seqs or other text columns)
            numeric_cols = M.select_dtypes(include=["number"]).columns
            M = M[numeric_cols]
            if args.method == 17:
                obj = FeatureExtractor.run("cross", args.method, matrix=M, k=getattr(args, "k", None))
            else:
                obj = FeatureExtractor.run("cross", args.method, matrix=M)
            save_output(obj, args.output)

        #NN class
        elif args.domain == "nn":
            try:
                FeatureExtractor._module("nn")
            except Exception as e:
                raise RuntimeError("NNFeatures is not available (import failed).") from e
            if args.list:
                print(FeatureExtractor.list_methods("nn"))
                return

            if args.seqs_csv:
                ids, seqs = load_sequences_csv(args.seqs_csv)
            else:
                seqs = load_sequences(args.seqs)
                ids = load_txt_list(args.sample_ids)
                if ids is not None and len(ids) != len(seqs):
                    raise ValueError(
                        f"sample_ids length ({len(ids)}) != sequences length ({len(seqs)})"
                    )

            ids2, seqs2 = preprocess_sequences(
                ids,
                seqs,
                to_upper=not args.no_upper,
                replace_t_with_u=not args.keep_t,
                strict=args.strict,
            )

            if args.save_clean:
                if ids2 is None:
                    ids2 = [f"s{i}" for i in range(len(seqs2))]
                pd.DataFrame({"id": ids2, "seq": seqs2}).to_csv(args.save_clean, index=False)
                print(f"[saved] cleaned sequences -> {args.save_clean}", file=sys.stderr)

            obj = FeatureExtractor.run(
                "nn",
                args.method,
                seqs2,
                return_format=args.return_format,
                sample_ids=ids2,
                batch_size=args.batch_size,
                layer=getattr(args, "layer", None),
                window=getattr(args, "window", None),
                stride=getattr(args, "stride", None),
                agg=getattr(args, "agg", None),
            )
            save_output(obj, args.output)

        # RNA-ALL convenience runner
        elif args.domain == "rna-all":
            props = self._load_props_csv(args.props_csv) if args.props_csv else None

            # Load sequences
            if args.seqs_csv:
                ids, seqs = load_sequences_csv(args.seqs_csv)
                stem = Path(args.seqs_csv).stem
            else:
                seqs = load_sequences(args.seqs)
                ids = load_txt_list(args.sample_ids)
                stem = Path(args.seqs).stem
                if ids is not None and len(ids) != len(seqs):
                    raise ValueError(
                        f"sample_ids length ({len(ids)}) != sequences length ({len(seqs)})"
                    )

            # Preprocess (drop invalid chars including N by default)
            ids2, seqs2 = preprocess_sequences(
                ids,
                seqs,
                to_upper=not args.no_upper,
                replace_t_with_u=not args.keep_t,
                strict=args.strict,
                valid_alphabet=set(ALPHABET),
            )

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            method_params = {
                1: {"k": args.k},
                2: {"k": args.k},
                3: {"lam": args.lam, "w": args.weight, "props": props},
                4: {"L": args.L, "props": props},
                5: {"L": args.L, "props": props},
                6: {"L": args.L, "props": props},
                7: {},
                8: {},  # di_composition uses k=2 internally
                9: {},  # tri_composition uses k=3 internally
                10: {},
                11: {"k_gap": args.k_gap},
                12: {"k_gap": args.k_gap},
            }

            for mid, name in sorted(RnaFeatures.METHOD_MAP.items()):
                kwargs = method_params.get(mid, {}).copy()
                kwargs.update(
                    {
                        "return_format": args.return_format,
                        "sample_ids": ids2,
                    }
                )
                # Only methods with composition/k-mer style outputs accept normalize.
                if mid in (1, 2, 7, 8, 9, 10, 11, 12):
                    kwargs["normalize"] = args.normalize
                if "props" in method_params.get(mid, {}) and props is None:
                    print(f"[skip] method {mid} ({name}) requires --props_csv", file=sys.stderr)
                    continue
                print(f"[run] method {mid} ({name})", file=sys.stderr)
                cols, df = FeatureExtractor.run("rna", mid, seqs2, **kwargs)
                out_path = outdir / f"{stem}_{name}.csv"
                if isinstance(df, pd.DataFrame):
                    df.to_csv(out_path, index=False)
                else:
                    pd.DataFrame(df, columns=cols).to_csv(out_path, index=False)
                print(f"[saved] {out_path}", file=sys.stderr)
