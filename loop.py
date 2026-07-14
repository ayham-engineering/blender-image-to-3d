"""The iterative render-critique-patch loop for a single (coder, critic) pair.

run() drives one benchmark trial: a critic model produces a spec from the
reference image, a coder model writes a Blender geometry script, the harness
renders it, the critic critiques the render against the reference, and the
coder patches. This repeats, hill-climbing on silhouette IoU loss, until it
plateaus, the critic reports no diffs, or max_iters is reached.

Three load-bearing rules (see inline comments):
  RULE 1 — a Blender traceback is never scored; it goes straight back to the
           coder and is logged as traceback=True, keeping crashes out of the
           score signal.
  RULE 2 — every non-crash patch is made from best_script (hill climbing),
           never from the last script (which would be a random walk).
  RULE 3 — return best_script, not the last one.

The model-calling functions are imported as module globals so a harness
(e.g. benchmark.py --dry-run) can monkeypatch loop.gen_spec / loop.gen_script
/ loop.gen_critique with free stubs.
"""

from __future__ import annotations

import tempfile
from math import inf
from pathlib import Path

from models import gen_spec, gen_script, gen_critique
from render_rig import render
from prepare_reference import prepare_reference, prepare_reference_silhouette
from score import score


def plateaued(log: list) -> bool:
    """True when the last two consecutive scored iterations each improved <2%.

    Only scored iterations (traceback=False) count. The objective is the
    silhouette IoU loss (lower is better), so improvement = (prev - cur) / prev.
    A regression (cur >= prev) counts as <2% improvement too, which is
    correctly treated as a plateau.
    """
    scored = [entry["iou_loss"] for entry in log if not entry["traceback"]]
    if len(scored) < 3:
        return False

    def improvement(prev: float, cur: float) -> float:
        if prev <= 0:
            return 0.0
        return (prev - cur) / prev

    a, b, c = scored[-3], scored[-2], scored[-1]
    return improvement(a, b) < 0.02 and improvement(b, c) < 0.02


def run(ref_image, coder_model: str, critic_model: str, max_iters: int = 5) -> dict:
    """Run one (coder, critic) benchmark trial. See module docstring for rules."""
    ref_image = Path(ref_image)
    workdir = Path(tempfile.mkdtemp(prefix="loop_"))

    tokens: list = []

    def record(model: str, usage: dict) -> None:
        tokens.append({"model": model, "in": usage["in"], "out": usage["out"]})

    def log_line(msg: str) -> None:
        print(f"  [{coder_model} vs {critic_model}] {msg}", flush=True)

    # Spec (critic) and reference grids (RGB + silhouette), computed once.
    spec, u = gen_spec(ref_image, critic_model)
    record(critic_model, u)
    ref_grid = prepare_reference(ref_image, workdir / "reference")
    ref_sil = prepare_reference_silhouette(ref_image, workdir / "reference")

    # First script (coder), from scratch.
    script, u = gen_script(spec, None, None, None, coder_model)
    record(coder_model, u)

    best_script, best_score, best_grid, best_sil = None, inf, None, None
    log: list = []

    for k in range(max_iters):
        script_path = workdir / f"script_{k}.py"
        script_path.write_text(script, encoding="utf-8")
        grid, sil, err = render(script_path, workdir / f"render_{k}")

        # RULE 1: a Blender traceback is NOT a score. Do not score it, do not
        # critique it, do not log it as inf. Log traceback=True and go straight
        # back to the coder with the verbatim traceback. traceback_count is the
        # signal for whether a model actually knows the bpy API; letting
        # failures leak into the score would destroy it.
        if err:
            log_line(f"iter {k}: traceback")
            script, u = gen_script(spec, script, [{"error": err}], None, coder_model)
            record(coder_model, u)
            log.append({"iter": k, "traceback": True})
            continue

        # Primary objective is silhouette IoU loss; LPIPS/SSIM/MSE are logged
        # only. Hill-climb on s (the IoU loss).
        s, metrics = score(ref_grid, grid, ref_sil, sil)
        log.append({"iter": k, "traceback": False, **metrics})
        log_line(f"iter {k}: IoU_loss={s:.4f} (IoU={metrics['iou']:.4f})")

        if s < best_score:
            best_score, best_script, best_grid, best_sil = s, script, grid, sil

        if plateaued(log):  # <2% improvement over 2 consecutive scored iters
            log_line(f"iter {k}: plateaued, stopping")
            break

        # Show the critic both shape (silhouettes) and shading (RGB) together.
        diffs, u = gen_critique(ref_grid, best_grid, ref_sil, best_sil, spec, critic_model)
        record(critic_model, u)
        if not diffs:
            log_line(f"iter {k}: critic reports no diffs, stopping")
            break

        # RULE 2: ALWAYS patch from best_script, NEVER the last script. This is
        # hill climbing; patching from the last script turns it into a random
        # walk that visibly degrades by ~iteration 4.
        script, u = gen_script(spec, best_script, diffs, best_grid, coder_model)
        record(coder_model, u)

    # RULE 3: return best_script, not the last one.
    return {
        "best_script": best_script,
        "best_score": best_score,
        "best_grid": best_grid,
        "best_sil": best_sil,
        "log": log,
        "traceback_count": sum(entry["traceback"] for entry in log),
        "tokens": tokens,
        "spec": spec,
    }
