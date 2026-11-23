"""HR assistant: RAG over Ukrainian labour law and the HR policies of a fictional company, with evals.

    python main.py all                  # the whole pipeline: ingest -> index -> eval-retrieval -> eval --check
    python main.py ingest               # data/raw -> data/processed (chunks.jsonl, manifest.json)
    python main.py index                # embeddings (cached in data/index/)
    python main.py ask "питання"        # answer with citations
    python main.py eval-retrieval       # recall@k / MRR for 7 search configurations
    python main.py eval --version v5    # end-to-end eval with deterministic scorers and LLM judges
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all", help="ingest, index, retrieval eval and end-to-end eval with thresholds")
    sub.add_parser("ingest", help="extract, clean, deduplicate and chunk the corpus")
    sub.add_parser("index", help="embed chunks (structure-aware and fixed baseline), cached in data/index/")
    s = sub.add_parser("search", help="one query against the index")
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid_rerank", choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    s.add_argument("--current-only", action="store_true", help="filter out outdated versions")
    s.add_argument("-k", type=int, default=5)
    sub.add_parser("eval-retrieval", help="recall@k / MRR on evals/retrieval_set.jsonl for several configurations")
    a = sub.add_parser("ask", help="answer one question with citations")
    a.add_argument("question")
    a.add_argument("--version", default=None, help="prompt/pipeline version (default: latest)")
    e = sub.add_parser("eval", help="end-to-end eval on evals/testset.jsonl")
    e.add_argument("--version", default=None)
    e.add_argument("--split", default="all", choices=["dev", "test", "devtest", "holdout", "all"])
    e.add_argument("--no-judge", action="store_true", help="deterministic scorers only")
    e.add_argument("--check", action="store_true", help="exit 1 if a metric is below evals/thresholds.json")
    sub.add_parser("label-sheet", help="write evals/human_labels.csv for manual labelling of answers")
    sub.add_parser("calibrate", help="agreement of the LLM judge with the human labels")
    args = ap.parse_args()
    if args.cmd == "all":
        from evals import evaluate, retrieval_eval
        from src.hr_rag import ingest, rag
        missing = [d.id for d in ingest.load_registry() if not d.path.exists()]
        if missing:   # a fresh clone has no hand-downloaded ILO PDFs (not redistributed): keep the committed chunks
            print(f"ingest skipped: {len(missing)} source file(s) not in data/raw ({', '.join(missing)}); "
                  "using the committed data/processed/ (see README, «Корпус»)")
        else:
            ingest.print_summary(ingest.run())
        retrieval_eval.run()                       # builds both indexes (embeddings come from the cache)
        evaluate.main(rag.DEFAULT, "all", True, True)
        return
    if args.cmd in ("ask", "eval"):
        from src.hr_rag import rag
        args.version = args.version or rag.DEFAULT
    if args.cmd == "ask":
        print(rag.render(rag.answer(args.question, args.version)))
    elif args.cmd == "eval":
        from evals import evaluate
        evaluate.main(args.version, args.split, not args.no_judge, args.check)
    elif args.cmd == "label-sheet":
        from evals import calibrate
        calibrate.write_sheet()
    elif args.cmd == "calibrate":
        from evals import calibrate
        calibrate.run()
    elif args.cmd == "eval-retrieval":
        from evals import retrieval_eval
        retrieval_eval.run()
    elif args.cmd == "ingest":
        from src.hr_rag import ingest
        ingest.print_summary(ingest.run())
    elif args.cmd == "index":
        from src.hr_rag.search import Index
        for f, name in (("chunks.jsonl", "structure"), ("chunks_fixed.jsonl", "fixed")):
            idx = Index(f, name)
            print(f"{name}: {len(idx.chunks)} chunks, vectors {idx.vectors.shape}")
    elif args.cmd == "search":
        from src.hr_rag.search import Index
        hits = Index().search(args.query, args.mode, args.k, {"is_current": True} if args.current_only else None)
        for h in hits:
            print(f"{h.rank:>2}. {h.score:8.4f}  {h.chunk['chunk_id']:<32} {h.chunk['header'][:90]}")


if __name__ == "__main__":
    main()
