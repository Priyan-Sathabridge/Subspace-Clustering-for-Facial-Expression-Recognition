"""Root orchestration CLI for the facial-expression clustering project.

Examples
--------
Run everything for both final methods:
    python pipeline.py --from 1 --method both

Resume from final coefficient construction:
    python pipeline.py --from 4 --method both

Run only affinity through clustering for Robust SSC:
    python pipeline.py --from 5 --to 6 --method robust_ssc

Run a single stage:
    python pipeline.py --only 10 --method robust_lrsc

Inspect readiness without executing:
    python pipeline.py --status
    python pipeline.py --from 4 --to 7 --method both --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pipeline_config as cfg


@dataclass(frozen=True)
class StageSpec:
    """Describe a script and whether it runs once or separately per model."""

    number: int
    key: str
    name: str
    directory: str
    script: str
    per_method: bool = False
    step4_multi_method: bool = False


STAGES = [
    StageSpec(1, "extract", "Geometry feature extraction", "step1_feature_extraction", "run.py"),
    StageSpec(2, "process", "Feature validation / processing", "step2_feature_processing", "run.py"),
    StageSpec(3, "pca", "Principal component analysis", "step3_pca", "run_pipeline.py"),
    StageSpec(4, "coefficients", "Final robust coefficient matrices", "step4_coefficients", "run.py", step4_multi_method=True),
    StageSpec(5, "affinity", "Affinity matrix construction", "step5_affinity", "run.py", per_method=True),
    StageSpec(6, "cluster", "Spectral clustering", "step6_clustering", "run.py", per_method=True),
    StageSpec(7, "visualize", "Final visualisations", "step7_visualization", "run.py", per_method=True),
    StageSpec(8, "evaluate", "Internal cluster evaluation", "step8_evaluation", "pipeline.py", per_method=True),
    StageSpec(9, "anova", "ANOVA feature analysis", "step9_anova", "run.py", per_method=True),
    StageSpec(10, "external", "External labelled evaluation", "step10_external_evaluation", "run.py", per_method=True),
]

ALIASES = {
    "1": 1, "extract": 1, "feature_extraction": 1,
    "2": 2, "process": 2, "processing": 2, "feature_processing": 2,
    "3": 3, "pca": 3,
    "4": 4, "coefficient": 4, "coefficients": 4,
    "5": 5, "affinity": 5,
    "6": 6, "cluster": 6, "clustering": 6,
    "7": 7, "visualize": 7, "visualization": 7,
    "8": 8, "evaluate": 8, "evaluation": 8, "internal_evaluation": 8,
    "9": 9, "anova": 9,
    "10": 10, "external": 10, "external_evaluation": 10,
}

STAGE_BY_NUMBER = {s.number: s for s in STAGES}


def resolve_stage(token: str | int) -> int:
    if isinstance(token, int):
        n = token
    else:
        key = str(token).strip().lower()
        if key not in ALIASES:
            valid = ", ".join(str(s.number) + "/" + s.key for s in STAGES)
            raise ValueError(f"Unknown stage '{token}'. Valid stages: {valid}")
        n = ALIASES[key]
    if n not in STAGE_BY_NUMBER:
        raise ValueError(f"Stage must be between 1 and {max(STAGE_BY_NUMBER)}.")
    return n


def selected_methods(value: str) -> list[str]:
    method = cfg.normalize_method(value)
    return list(cfg.FINAL_METHODS) if method == "both" else [method]


def prerequisite_paths(stage: int, method: str | None = None) -> list[Path]:
    """List required paths; stage scripts perform content and schema validation."""
    if stage == 1:
        return [cfg.IMAGE_FOLDER]
    if stage == 2:
        return [cfg.geometry_features_csv()]
    if stage == 3:
        return [cfg.STEP2_PROCESSED_FEATURES_CSV]
    if stage == 4:
        return [cfg.PCA_FEATURES_CSV]
    if method is None:
        raise ValueError(f"Stage {stage} requires a method.")
    if stage == 5:
        return [cfg.coefficient_csv(method)]
    if stage == 6:
        return [cfg.affinity_csv(method), cfg.PCA_FEATURES_CSV]
    if stage == 7:
        return [cfg.labels_csv(method), cfg.affinity_csv(method), cfg.PCA_FEATURES_CSV, cfg.IMAGE_FOLDER]
    if stage == 8:
        return [cfg.labels_csv(method), cfg.affinity_csv(method), cfg.PCA_FEATURES_CSV]
    if stage == 9:
        return [cfg.labels_csv(method), cfg.ANOVA_FEATURES_CSV]
    if stage == 10:
        return [cfg.labels_csv(method), cfg.GROUND_TRUTH_CSV]
    return []


def expected_outputs(stage: int, method: str | None = None) -> list[Path]:
    if stage == 1:
        return [cfg.STEP1_GEOMETRY_FEATURES_CSV]
    if stage == 2:
        return [cfg.STEP2_PROCESSED_FEATURES_CSV]
    if stage == 3:
        return [cfg.PCA_FEATURES_CSV]
    if stage == 4:
        if method is None:
            return []
        return [cfg.coefficient_csv(method), cfg.error_matrix_csv(method)]
    if method is None:
        return []
    if stage == 5:
        return [cfg.affinity_csv(method)]
    if stage == 6:
        return [cfg.labels_csv(method)]
    if stage == 7:
        return [cfg.visualization_dir(method)]
    if stage == 8:
        return [cfg.evaluation_results_csv(method)]
    if stage == 9:
        return [cfg.anova_csv(method)]
    if stage == 10:
        return [cfg.external_metrics_csv(method)]
    return []


def path_ready(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        try:
            next(path.iterdir())
            return True
        except StopIteration:
            return False
    return False


def check_prerequisites(stage: int, method: str | None) -> list[Path]:
    return [p for p in prerequisite_paths(stage, method) if not p.exists()]


def is_complete(stage: int, method: str | None) -> bool:
    """Check artifact presence only, not convergence or configuration freshness."""
    outputs = expected_outputs(stage, method)
    return bool(outputs) and all(path_ready(p) for p in outputs)


def stage_script(spec: StageSpec) -> Path:
    return cfg.PROJECT_ROOT / spec.directory / spec.script


def run_process(spec: StageSpec, method_env: str, log_path: Path, dry_run: bool) -> int:
    script = stage_script(spec)
    cmd = [sys.executable, script.name]
    cwd = script.parent

    # A separate process and stage-local cwd prevent unrelated config.py modules
    # from sharing an import cache; PYTHONPATH exposes the central configuration.
    env = os.environ.copy()
    env[cfg.METHOD_ENV] = method_env
    env["SC_PIPELINE_ROOT"] = str(cfg.PROJECT_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cfg.PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    label = f"Step {spec.number} - {spec.name}"
    if spec.per_method or spec.step4_multi_method:
        label += f" [{method_env}]"

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print(f"cwd     : {cwd}")
    print(f"command : {' '.join(cmd)}")

    if dry_run:
        print("DRY RUN: not executed")
        return 0

    if not script.exists():
        print(f"ERROR: stage script does not exist: {script}")
        return 2

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"{label}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"command: {' '.join(cmd)}\n\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def stage_instances(spec: StageSpec, methods: list[str]) -> list[tuple[str | None, str]]:
    """Pair each path branch with the method selector sent to its subprocess."""
    if spec.step4_multi_method:
        env_value = "both" if len(methods) == 2 else methods[0]
        return [(None, env_value)]
    if spec.per_method:
        return [(m, m) for m in methods]
    return [(None, "both")]


def print_stage_list() -> None:
    print("Pipeline stages")
    print("---------------")
    for s in STAGES:
        branch = "method-specific" if s.per_method else ("multi-method" if s.step4_multi_method else "shared")
        print(f"{s.number:>2}. {s.key:<12} {s.name:<38} [{branch}]")


def print_status(methods: list[str]) -> None:
    print(f"Data root: {cfg.DATA_ROOT}")
    print(f"Methods  : {', '.join(methods)}")
    print()
    print(f"{'Stage':<7} {'Method':<14} {'Complete':<10} Expected output")
    print("-" * 90)
    for s in STAGES:
        instances = methods if (s.per_method or s.number == 4) else [None]
        for method in instances:
            complete = is_complete(s.number, method)
            outs = expected_outputs(s.number, method)
            out_text = ", ".join(str(p) for p in outs) if outs else "-"
            print(
                f"{s.number:<7} {(method or 'shared'):<14} "
                f"{str(complete):<10} {out_text}"
            )


def select_stage_numbers(args: argparse.Namespace) -> list[int]:
    if args.only:
        values = [x.strip() for x in args.only.split(",") if x.strip()]
        return sorted(dict.fromkeys(resolve_stage(v) for v in values))

    if args.legacy_stage and not args.from_stage and not args.to_stage:
        if args.legacy_stage.lower() == "all":
            return [s.number for s in STAGES]
        return [resolve_stage(args.legacy_stage)]

    start = resolve_stage(args.from_stage or "1")
    end = resolve_stage(args.to_stage or str(max(STAGE_BY_NUMBER)))
    if end < start:
        raise ValueError("--to must not be earlier than --from.")
    return list(range(start, end + 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the facial-expression clustering pipeline from any stage."
    )
    p.add_argument(
        "legacy_stage",
        nargs="?",
        help="Backward-compatible single stage alias or 'all'. Prefer --from/--only.",
    )
    p.add_argument("--from", dest="from_stage", help="First stage number or name.")
    p.add_argument("--to", dest="to_stage", help="Last stage number or name.")
    p.add_argument("--only", help="Comma-separated stages to run, e.g. 4,5,6.")
    p.add_argument(
        "--method",
        default="both",
        choices=["both", "robust_ssc", "robust_lrsc", "ssc", "lrsc"],
        help="Algorithm branch for stages 4-10.",
    )
    p.add_argument("--list", action="store_true", help="List available stages and exit.")
    p.add_argument("--status", action="store_true", help="Show output/readiness status and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--skip-completed", action="store_true", help="Skip stages whose expected outputs already exist.")
    p.add_argument("--no-checks", action="store_true", help="Do not validate input prerequisites before each stage.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue to later stages after a failure.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    methods = selected_methods(args.method)

    if args.list:
        print_stage_list()
        return 0
    if args.status:
        print_status(methods)
        return 0

    try:
        stage_numbers = select_stage_numbers(args)
    except ValueError as exc:
        parser.error(str(exc))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = cfg.PIPELINE_LOG_ROOT / run_id
    manifest = {
        "run_id": run_id,
        "project_root": str(cfg.PROJECT_ROOT),
        "data_root": str(cfg.DATA_ROOT),
        "methods": methods,
        "stages": stage_numbers,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "records": [],
    }

    failures = 0

    for stage_no in stage_numbers:
        spec = STAGE_BY_NUMBER[stage_no]
        for method_for_paths, method_env in stage_instances(spec, methods):
            # Step 4 produces one or both methods in one process; check every requested output.
            output_methods = methods if spec.number == 4 else ([method_for_paths] if method_for_paths else [None])

            if args.skip_completed:
                if spec.number == 4:
                    done = all(is_complete(4, m) for m in methods)
                else:
                    done = is_complete(stage_no, method_for_paths)
                if done:
                    print(f"Skipping completed Step {stage_no} {method_for_paths or ''}".rstrip())
                    continue

            # Dry runs also validate paths unless explicitly disabled: they do
            # not create outputs needed by later stages in the preview.
            if not args.no_checks:
                if spec.number == 4:
                    missing = check_prerequisites(4, None)
                else:
                    missing = check_prerequisites(stage_no, method_for_paths)
                if missing:
                    print("\nCannot start stage because prerequisite files are missing:")
                    for path in missing:
                        print(f"  - {path}")
                    print("Run the preceding stage, correct DATA_ROOT, or use --no-checks if intentional.")
                    return 3

            suffix = method_env if (spec.per_method or spec.step4_multi_method) else "shared"
            log_path = run_dir / f"step_{stage_no:02d}_{suffix}.log"
            started = datetime.now()
            code = run_process(spec, method_env, log_path, args.dry_run)
            ended = datetime.now()

            manifest["records"].append(
                {
                    "stage": stage_no,
                    "key": spec.key,
                    "method": method_env,
                    "return_code": code,
                    "started_at": started.isoformat(timespec="seconds"),
                    "ended_at": ended.isoformat(timespec="seconds"),
                    "log": str(log_path),
                }
            )

            if code != 0:
                failures += 1
                print(f"\nStage failed with return code {code}: Step {stage_no} {method_env}")
                if not args.continue_on_error:
                    if not args.dry_run:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
                        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    return code

    manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nRun manifest: {run_dir / 'manifest.json'}")

    print("\nPipeline selection complete.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
