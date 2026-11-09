"""Command-line interface for the AI Resume Screening & Fairness Assessment system."""

from __future__ import annotations

import argparse
import json
import sys

from .llm import GeminiProvider
from .models import Config, load_benchmark_corpus
from .report import run_pipeline, validate_auditor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frs", description="AI Resume Screening & Fairness Assessment CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pipe = sub.add_parser("pipeline", help="Run the counterfactual fairness audit and mitigation pipeline")
    p_pipe.add_argument("--provider", choices=["gemini", "mock"], default="gemini")
    p_pipe.add_argument("--max-resumes", type=int, default=3)
    p_pipe.add_argument("--max-variants", type=int, default=12)
    p_pipe.add_argument("--run-name", default="audit")
    p_pipe.add_argument("--null-bias", action="store_true", help="Use zero bias in mock mode")

    sub.add_parser("validate-auditor", help="Run the 5-check Null vs Injected bias validation oracle")

    p_web = sub.add_parser("web", help="Launch the FastAPI single-resume web evaluator")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)

    sub.add_parser("doctor", help="Check dataset, environment, and Gemini readiness")

    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        cfg = Config(max_resumes=None, max_jobs=None, max_variants_per_resume=None)
        r, j, p, v = load_benchmark_corpus(cfg)
        gem = GeminiProvider(model=cfg.model)
        info = {
            "status": "ok",
            "resumes": len(r),
            "jobs": len(j),
            "pairs": len(p),
            "variants": len(v),
            "gemini_ready": gem.is_ready(),
            "model": cfg.model,
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.cmd == "validate-auditor":
        res = validate_auditor()
        print(json.dumps(res, indent=2))
        return 0 if res["passed"] else 1

    if args.cmd == "pipeline":
        demo_bias = {} if (args.provider != "mock" or args.null_bias) else {"gender": {"female": -3.0}, "socioeconomic": {"low": -3.0}}
        cfg = Config(
            run_name=args.run_name,
            provider=args.provider,
            max_resumes=args.max_resumes if args.max_resumes > 0 else None,
            max_variants_per_resume=args.max_variants if args.max_variants > 0 else None,
            mock_bias=demo_bias,
        )
        report, run_dir = run_pipeline(cfg)
        print(json.dumps({
            "run_id": report.run_id,
            "run_dir": str(run_dir),
            "report_html": str(run_dir / "report.html"),
            "n_scores": report.n_scores,
            "headline_findings": report.headline_findings,
        }, indent=2))
        return 0

    if args.cmd == "web":
        import uvicorn

        uvicorn.run("frs.web_api:app", host=args.host, port=args.port, reload=False)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

