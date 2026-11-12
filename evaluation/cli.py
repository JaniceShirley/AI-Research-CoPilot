"""Command line entry point: ``python -m evaluation {run,validate} ...``"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys

from .dataset import DatasetError, load_dataset, resolve_paper_files, select_items

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"


def _load_env(env_file: str | None) -> None:
    from dotenv import load_dotenv
    candidates = [env_file] if env_file else [os.path.join(REPO_ROOT, ".env"),
                                             os.path.join(BACKEND_DIR, ".env")]
    for c in candidates:
        if c and os.path.isfile(c):
            load_dotenv(c, override=False)
    if env_file and not os.path.isfile(env_file):
        sys.exit(f"--env-file not found: {env_file}")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Put it in .env at the repo root or backend/, "
                 "pass --env-file, or export it. (The key is never printed or written to results.)")


def _import_backend():
    """Import the production modules exactly as the FastAPI app does."""
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    import paper_manager  # noqa: E402
    import rag_pipeline  # noqa: E402
    return paper_manager, rag_pipeline


def _git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True,
                              text=True, check=False).stdout.strip()
    return {"git_commit": git("rev-parse", "--short", "HEAD") or "unknown",
            "git_dirty": bool(git("status", "--porcelain", "--", "backend"))}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="Path to the evaluation dataset JSON")
    p.add_argument("--papers-dir", required=True,
                   help="Folder containing the PDFs named in the dataset (read-only)")
    p.add_argument("--env-file", help="Path to a .env with GROQ_API_KEY (default: repo .env or backend/.env)")
    p.add_argument("--match-threshold", type=float, default=0.6,
                   help="Min share of an evidence quote's word trigrams a chunk must contain (default 0.6)")
    p.add_argument("--ids", nargs="*", help="Only evaluate these item ids")
    p.add_argument("--limit", type=int, help="Only evaluate the first N items")


def cmd_validate(args) -> int:
    data = load_dataset(args.dataset)
    items = select_items(data, args.ids, args.limit)
    pdfs = resolve_paper_files(data, args.papers_dir)
    _load_env(args.env_file)
    paper_manager, pipeline = _import_backend()
    from .runner import check_evidence, index_chunks
    from .workspace import ingest_papers, isolated_workspace, new_eval_session_id
    sid = new_eval_session_id()
    with isolated_workspace(paper_manager, pipeline):
        ingest_papers(paper_manager, pipeline, sid, pdfs)
        chunks = index_chunks(pipeline, sid)
    report = check_evidence(items, chunks, args.match_threshold)
    bad = [r for r in report if not r["ok"]]
    print(f"{len(chunks)} chunks indexed from {len(pdfs)} paper(s).")
    for r in report:
        mark = "ok " if r["ok"] else "BAD"
        print(f"  [{mark}] {r['item_id']} source {r['source_index']} "
              f"({r['paper']} p.{r['page']}): coverage {r['best_coverage']}")
    print(f"{len(report) - len(bad)}/{len(report)} evidence quotes found in the index.")
    return 1 if bad else 0


def cmd_run(args) -> int:
    data = load_dataset(args.dataset)
    items = select_items(data, args.ids, args.limit)
    pdfs = resolve_paper_files(data, args.papers_dir)
    if any(k <= 0 for k in args.k):
        sys.exit("--k values must be positive")
    _load_env(args.env_file)
    paper_manager, pipeline = _import_backend()

    from .judge import LLMJudge
    from .report import write_all
    from .runner import RunConfig, run_evaluation, slugify

    judge = None
    if not args.no_judge:
        from groq import Groq
        # A separate client: judge calls are never mixed into pipeline timings.
        judge = LLMJudge(Groq(api_key=os.getenv("GROQ_API_KEY")), args.judge_model,
                         max_tokens=args.judge_max_tokens)

    embed = None if args.no_embedding_relevance else pipeline.embedding_model.embed_query
    production_model = getattr(pipeline, "GROQ_MODEL", "unknown")
    if args.generation_model:
        print(f"NOTE: generation model overridden for this run only: "
              f"{production_model} -> {args.generation_model}")
        pipeline.GROQ_MODEL = args.generation_model
    cfg = RunConfig(
        k_values=sorted(set(args.k)),
        match_threshold=args.match_threshold,
        retries=args.retries,
        retry_wait_s=args.retry_wait,
        sleep_between_s=args.sleep,
        keep_workspace=args.keep_workspace,
        echo_pipeline_logs=args.verbose,
        extra={
            **_git_info(),
            "generation_model": getattr(pipeline, "GROQ_MODEL", "unknown"),
            "production_generation_model": production_model,
            "generation_model_overridden": bool(args.generation_model),
            "embedding_model": getattr(pipeline.embedding_model, "model_name", "unknown"),
        },
    )
    try:
        result = run_evaluation(data, items, pdfs, paper_manager, pipeline, pipeline.FAISS, cfg,
                                judge=judge, embed=embed)
    finally:
        pipeline.GROQ_MODEL = production_model

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.output_dir, f"{stamp}-{slugify(data.get('name'))}")
    paths = write_all(result, out_dir)
    s = result["summary"]
    print("\nDone.")
    print(f"  questions: {s['n_questions']}  answered: {s['n_answered']}  "
          f"no-LLM: {s['n_no_generation']}  errors: {s['n_errors']}")
    for k in cfg.k_values:
        m = s[f"recall@{k}"]["mean"]
        print(f"  Recall@{k}: {'n/a' if m is None else f'{m:.3f}'}")
    for name in ("faithfulness_llm_judge", "answer_relevance_llm_judge",
                 "latency_retrieval_s", "latency_generation_s", "latency_total_s"):
        m = s[name]["mean"]
        print(f"  {name}: {'n/a' if m is None else f'{m:.3f}'}")
    print(f"  data integrity unchanged: {result['integrity']['unchanged']}")
    for kind, p in paths.items():
        print(f"  {kind}: {p}")
    return 0 if s["n_errors"] == 0 else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation",
                                     description="Evaluate the AI Research Co-Pilot RAG pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Index the papers (no LLM calls) and check every evidence "
                                        "quote in the dataset can be found in a chunk")
    _common(v)
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="Run the full evaluation")
    _common(r)
    r.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="K values for Recall@K")
    r.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
                   help=f"Groq model used as LLM judge (default {DEFAULT_JUDGE_MODEL}, "
                        "or $EVAL_JUDGE_MODEL)")
    r.add_argument("--generation-model",
                   help="EVALUATION ONLY: temporarily replace rag_pipeline.GROQ_MODEL for this run "
                        "(production code and config are not changed; the report flags the override)")
    r.add_argument("--judge-max-tokens", type=int,
                   help="Cap on judge output tokens (needed when the judge model's per-minute "
                        "output-token limit is small, e.g. 900 for a 1000 OTPM limit)")
    r.add_argument("--no-judge", action="store_true",
                   help="Skip LLM-judge faithfulness/relevance (deterministic metrics only)")
    r.add_argument("--no-embedding-relevance", action="store_true")
    r.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "evaluation", "results"))
    r.add_argument("--retries", type=int, default=2, help="Retries for rate-limited questions")
    r.add_argument("--retry-wait", type=float, default=20.0, help="Seconds to wait before a retry")
    r.add_argument("--sleep", type=float, default=0.0, help="Seconds to pause between questions")
    r.add_argument("--keep-workspace", action="store_true",
                   help="Keep the temporary workspace (uploads + FAISS index) for inspection")
    r.add_argument("--verbose", action="store_true", help="Echo the pipeline's own log lines")
    r.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DatasetError as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1
