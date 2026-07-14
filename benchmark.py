"""Benchmark harness: run the render-critique-patch loop for several
(coder, critic) model pairs against the same reference image, then report
final silhouette IoU loss, iteration counts, traceback counts, and USD cost
per pair — plus an empty-scene baseline row (the "produced nothing" floor).

Usage:
    python benchmark.py --ref path/to/reference.png
    python benchmark.py --ref path/to/reference.png --max-iters 5
    python benchmark.py --dry-run      # free: stubs the paid model calls

A model must never both code and critique in the same pair — a model grading
its own output rubber-stamps it. This is asserted before any pair runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from math import inf
from pathlib import Path

import loop


# Per-MTok pricing, (input_rate, output_rate) in USD. One place to edit.
RATES = {
    "claude-sonnet-5": (2, 10),
    "claude-opus-4-8": (5, 25),
    "claude-fable-5": (10, 50),
}

# Default (coder, critic) pairs. Coder != critic in every pair.
DEFAULT_PAIRS = [
    ("claude-sonnet-5", "claude-opus-4-8"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("claude-fable-5", "claude-sonnet-5"),
]


def cost_usd(tokens: list) -> float:
    """Sum USD cost across every recorded usage entry using RATES."""
    total = 0.0
    for entry in tokens:
        model = entry["model"]
        if model not in RATES:
            raise KeyError(f"No rate defined for model '{model}' in RATES")
        rate_in, rate_out = RATES[model]
        total += entry["in"] / 1_000_000 * rate_in
        total += entry["out"] / 1_000_000 * rate_out
    return total


# --------------------------------------------------------------------------
# Dry-run stubs: replace the paid API calls with free hardcoded responses so
# the whole harness (render, score, grid assembly, table, file output) can be
# exercised end-to-end for zero cost. render() and score() run for real.
# --------------------------------------------------------------------------

_STUB_SPEC = {
    "object": "cube",
    "parts": [
        {
            "name": "body",
            "primitive": "cube",
            "approx_dims": [2, 2, 2],
            "position": [0, 0, 0],
            "rotation_euler": [0, 0, 0],
            "notes": "single test cube",
        }
    ],
    "relations": [],
    "overall_scale": 2.0,
}

_STUB_SCRIPT = "import bpy\nbpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))\n"

_STUB_USAGE = {"in": 100, "out": 100}


def _install_dry_run_stubs() -> None:
    loop.gen_spec = lambda ref_image, model: (_STUB_SPEC, dict(_STUB_USAGE))
    loop.gen_script = lambda spec, prev, diffs, grid, model: (_STUB_SCRIPT, dict(_STUB_USAGE))
    # Empty diffs -> the loop stops after the first scored iteration. Signature
    # matches gen_critique: (ref_grid, render_grid, ref_sil, render_sil, spec, model).
    loop.gen_critique = lambda ref_grid, render_grid, ref_sil, render_sil, spec, model: ([], dict(_STUB_USAGE))


def _make_placeholder_ref(out_dir: Path) -> Path:
    """Create a placeholder reference image (dark shape on light bg) for dry-run.

    A distinct foreground shape (not a flat fill) is needed so silhouette
    segmentation produces a real, non-degenerate mask.
    """
    from PIL import Image, ImageDraw

    path = out_dir / "placeholder_ref.png"
    img = Image.new("RGB", (256, 256), color=(210, 210, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 60, 176, 200], fill=(40, 40, 40))
    img.save(path)
    return path


def compute_baseline(ref_image: Path, out_dir: Path):
    """Score an empty scene (no geometry) against the reference.

    This is the "produced nothing" floor: an empty render has an all-white
    silhouette, so its IoU with the reference is 0 and its loss is ~1.0. If a
    model's best score barely beats this, the loop isn't working and the
    ranking is noise. Returns (loss, metrics) or None if the render failed.
    """
    from render_rig import render
    from prepare_reference import prepare_reference, prepare_reference_silhouette
    from score import score

    empty_script = out_dir / "_empty_scene.py"
    empty_script.write_text("import bpy\n# empty scene: no geometry\n", encoding="utf-8")

    grid, sil, err = render(empty_script, out_dir / "_baseline_render")
    if err:
        print(f"[baseline] empty-scene render failed:\n{err}", flush=True)
        return None

    ref_grid = prepare_reference(ref_image, out_dir / "_baseline_ref")
    ref_sil = prepare_reference_silhouette(ref_image, out_dir / "_baseline_ref")
    loss, metrics = score(ref_grid, grid, ref_sil, sil)
    return loss, metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, default=None, help="Path to the reference image.")
    parser.add_argument("--max-iters", type=int, default=5, help="Max loop iterations per pair.")
    parser.add_argument("--dry-run", action="store_true", help="Stub paid model calls; free end-to-end run.")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_out"), help="Where to write results.")
    args = parser.parse_args()

    # Coder must never equal critic in any pair (a model grading its own work
    # rubber-stamps it). Assert before doing any work.
    for coder, critic in DEFAULT_PAIRS:
        assert coder != critic, f"coder and critic must differ, got both = {coder}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _install_dry_run_stubs()

    ref_image = args.ref
    if ref_image is None:
        if args.dry_run:
            ref_image = _make_placeholder_ref(Path(tempfile.mkdtemp(prefix="bench_ref_")))
            print(f"[dry-run] no --ref given, using placeholder: {ref_image}", flush=True)
        else:
            parser.error("--ref is required unless --dry-run is used")
    ref_image = Path(ref_image)
    if not ref_image.is_file():
        parser.error(f"--ref does not exist: {ref_image}")

    # Empty-scene baseline: the "produced nothing" floor. Computed first so
    # every pair's score can be read against it.
    print("\n=== Computing empty-scene baseline (floor) ===", flush=True)
    baseline = compute_baseline(ref_image, args.out_dir)
    if baseline is not None:
        baseline_loss, baseline_metrics = baseline
        print(f"--- baseline IoU_loss={baseline_loss:.4f} (IoU={baseline_metrics['iou']:.4f})", flush=True)
    else:
        baseline_loss = None
        print("--- baseline unavailable (empty-scene render failed)", flush=True)

    results = []
    for coder, critic in DEFAULT_PAIRS:
        print(f"\n=== Running coder={coder} critic={critic} ===", flush=True)
        result = loop.run(ref_image, coder, critic, max_iters=args.max_iters)

        best_score = result["best_score"]
        n_iters = len(result["log"])
        tracebacks = result["traceback_count"]
        cost = cost_usd(result["tokens"])

        # Save best grid PNG and winning script (guard against all-traceback runs).
        best_grid = result["best_grid"]
        if best_grid is not None:
            shutil.copyfile(best_grid, args.out_dir / f"best_{coder}.png")
        best_sil = result.get("best_sil")
        if best_sil is not None:
            shutil.copyfile(best_sil, args.out_dir / f"best_{coder}_silhouette.png")
        best_script = result["best_script"]
        if best_script is not None:
            (args.out_dir / f"best_{coder}.py").write_text(best_script, encoding="utf-8")

        results.append(
            {
                "coder": coder,
                "critic": critic,
                "best_score": best_score,
                "iters": n_iters,
                "tracebacks": tracebacks,
                "cost_usd": cost,
                "log": result["log"],
                "tokens": result["tokens"],
                "spec": result["spec"],
            }
        )

        score_str = "FAIL" if best_score == inf else f"{best_score:.4f}"
        print(f"--- done: IoU_loss={score_str} iters={n_iters} tracebacks={tracebacks} cost=${cost:.4f}", flush=True)

    # results.json with the full log (including the baseline floor).
    results_path = args.out_dir / "results.json"
    results_path.write_text(
        json.dumps({"baseline_loss": baseline_loss, "pairs": results}, indent=2),
        encoding="utf-8",
    )

    # Markdown table. The objective is silhouette IoU loss (lower is better).
    print("\n## Benchmark results (primary metric: silhouette IoU loss, lower is better)\n")
    print("| coder | critic | final IoU loss | iters | tracebacks | cost USD |")
    print("| --- | --- | --- | --- | --- | --- |")
    baseline_str = "n/a" if baseline_loss is None else f"{baseline_loss:.4f}"
    print(f"| _(baseline: empty scene)_ | — | {baseline_str} | 0 | 0 | $0.0000 |")
    for r in results:
        score_str = "FAIL" if r["best_score"] == inf else f"{r['best_score']:.4f}"
        print(
            f"| {r['coder']} | {r['critic']} | {score_str} | "
            f"{r['iters']} | {r['tracebacks']} | ${r['cost_usd']:.4f} |"
        )
    print(f"\nWrote {results_path} and best_<coder>.png / .py / _silhouette.png to {args.out_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
