"""HR assistant: RAG over Ukrainian labour law and the HR policies of a fictional company, with evals.

    python main.py ingest      # data/raw -> data/processed (chunks.jsonl, manifest.json)
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="extract, clean, deduplicate and chunk the corpus")
    args = ap.parse_args()
    if args.cmd == "ingest":
        from src.hr_rag import ingest
        ingest.print_summary(ingest.run())


if __name__ == "__main__":
    main()
