import argparse
import pandas as pd
import sys
from typing import Dict
from loader import (
    load_sequences,
    load_txt_list,
    load_csv_df,
    load_edges_child_parent,
    load_sequences_csv,
    preprocess_sequences,
    save_output,
)
from feature_extractor import FeatureExtractor


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
        rna.add_argument("--k", type=int)
        rna.add_argument("--lam", type=int)
        rna.add_argument("--weight", type=float)
        rna.add_argument("--normalize", type=lambda s: s.lower() != "false", default=True)
        rna.add_argument("--return_format", choices=["matrix", "dataframe"], default="dataframe")

        # optional preprocessing flags
        rna.add_argument("--no_upper", action="store_true", help="Do not uppercase sequences")
        rna.add_argument("--keep_t", action="store_true", help="Do not convert T->U")
        rna.add_argument("--strict", action="store_true", help="Error on invalid chars instead of dropping")
        rna.add_argument("--save_clean", help="Optional path to save cleaned sequences as CSV (id,seq)")

        rna.add_argument("-o", "--output", help="Path to save output")

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

        self.parser = p


    def run(self, argv=None):
        args = self.parser.parse_args(argv)

        # List available methods
        if getattr(args, "list", False):
            print(FeatureExtractor.list_methods(args.domain))
            return

        #RNA class
        if args.domain == "rna":
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
            obj = FeatureExtractor.run(
                "rna",
                args.method,
                seqs2,
                k=args.k,
                lam=args.lam,
                w=args.weight,
                normalize=args.normalize,
                return_format=args.return_format,
                sample_ids=ids2,
            )
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
                    d: list(g["term"].astype(str)) for d, g in dtt.groupby("disease")
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
            obj = FeatureExtractor.run("cross", args.method, matrix=M, k=getattr(args, "k", None))
            save_output(obj, args.output)
