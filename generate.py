"""generate.py — one asset-generation pipeline with structured cross-model feedback.

Text and image inputs flow through the same loop. The output is always a
STYLIZED LOW-POLY GAME ASSET (procedural geometry, palette materials, connected
parts) exported as GLB — never a photo reconstruction.

    generate(out_glb, prompt=..., style_image=..., iters=3,
             coder="claude-opus-4-8", critic="claude-sonnet-5",
             director="claude-sonnet-5")

Inputs (at least one required):
    prompt only       -> the art director writes the spec from text
    style_image only  -> the art director writes the spec from the image AS
                         STYLE INSPIRATION (stylized asset, not pixel copy)
    both              -> the image guides style, the text adds direction

The loop:
    spec   = gen_spec_from_prompt(prompt, director, style_image)
    script = gen_script(spec, ..., coder, mode="asset")
    for i in range(iters):
        render a preview (RGB 6-view grid)
        if it crashed  -> send the verbatim traceback back to the coder, retry
        diffs = gen_asset_critique(preview, spec, critic, style_image)
        if no diffs    -> done
        script = gen_script(spec, script, diffs, preview, coder, mode="asset")
    export GLB with materials  (from best_script — see below)

Only a script that has actually rendered is ever exported, so a crash in the
final unvalidated revision can't fail the whole run. If nothing ever rendered,
that's a total failure and raises with the last traceback.

But "rendered" only means "didn't crash" — it does NOT mean "better". A critique
round can make the asset worse, and with no reference image there is nothing to
score against, so the loop cannot rank iterations. Instead it keeps every
successful iteration as a candidate, writes a labeled contact sheet of all their
previews next to the .glb, and lists them. The most recent is the DEFAULT, not a
claim that it is best; pass --pick (CLI) or choose=... (API) to select one.

The critic MUST be a different model from the coder — a model critiquing its own
output rubber-stamps it. This is asserted, not merely recommended.

Usage:
    python generate.py --prompt "low-poly palm tree, stylized, game-ready" --out palm.glb --iters 3
    python generate.py --style-image ref.jpg --out asset.glb
    python generate.py --style-image ref.jpg --prompt "make it autumnal" --out asset.glb

Reuses render_rig's Blender-binary resolution, traceback scanning, and the
geometry-only wrapper approach (clear scene, exec script). The final Blender
step exports GLB instead of rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

from render_rig import find_blender_binary, _extract_traceback, render
from models import gen_spec_from_prompt, gen_script, gen_asset_critique

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None


DEFAULT_CODER = "claude-opus-4-8"
DEFAULT_CRITIC = "claude-sonnet-5"
DEFAULT_DIRECTOR = "claude-sonnet-5"


# --------------------------------------------------------------------------
# Blender wrapper: clear scene, exec generated script, export GLB
# --------------------------------------------------------------------------

_EXPORT_WRAPPER_TEMPLATE = '''
import bpy
import traceback
import sys

GEOMETRY_SCRIPT_PATH = {geometry_script_path!r}
GLB_OUT_PATH = {glb_out_path!r}


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def exec_geometry_script():
    with open(GEOMETRY_SCRIPT_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    namespace = {{"__name__": "__geometry__", "bpy": bpy}}
    exec(compile(source, GEOMETRY_SCRIPT_PATH, "exec"), namespace)


def export_glb():
    # Merge the parts into the file as-is (all objects in the scene). Apply
    # modifiers, use +Y up (Three.js / glTF convention), and carry materials
    # through so the asset's palette survives into the .glb.
    bpy.ops.export_scene.gltf(
        filepath=GLB_OUT_PATH,
        export_format="GLB",
        export_apply=True,          # apply modifiers on export
        export_yup=True,            # +Y up
        export_materials="EXPORT",  # keep palette materials
    )


def main():
    clear_scene()
    exec_geometry_script()
    export_glb()


try:
    main()
except Exception:
    traceback.print_exc(file=sys.stderr)
    raise
'''


def _write_export_wrapper(geometry_script_path: Path, glb_out_path: Path) -> Path:
    wrapper_source = _EXPORT_WRAPPER_TEMPLATE.format(
        geometry_script_path=str(geometry_script_path.resolve()),
        glb_out_path=str(glb_out_path.resolve()),
    )
    fd, wrapper_path_str = tempfile.mkstemp(suffix="_export_wrapper.py", text=True)
    wrapper_path = Path(wrapper_path_str)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(wrapper_source)
    return wrapper_path


def export_glb(script_path: Path, glb_out_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Run the generated script in Blender headless and export it to GLB.

    Returns (glb_path, None) on success or (None, error_str) on failure.
    Never raises for Blender/subprocess errors — they are captured and returned.
    """
    script_path = Path(script_path)
    glb_out_path = Path(glb_out_path)
    glb_out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        blender_bin = find_blender_binary()
    except FileNotFoundError as e:
        return None, str(e)

    if glb_out_path.exists():
        glb_out_path.unlink()

    try:
        wrapper_path = _write_export_wrapper(script_path, glb_out_path)
    except OSError as e:
        return None, f"Failed to write export wrapper script: {e}"

    try:
        proc = subprocess.run(
            [str(blender_bin), "-b", "-P", str(wrapper_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        return None, f"Blender export timed out after 600s.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
    except OSError as e:
        return None, f"Failed to launch Blender binary '{blender_bin}': {e}"
    finally:
        try:
            wrapper_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Blender exits 0 even when the script raises; detect failure via stderr.
    traceback_text = _extract_traceback(proc.stderr or "")
    if traceback_text is not None:
        return None, traceback_text

    if not glb_out_path.exists():
        return None, (
            "Blender exited without producing a GLB file and no traceback was "
            "found in stderr. Raw stderr:\n" + (proc.stderr or "(empty)")
        )

    return glb_out_path, None


# --------------------------------------------------------------------------
# Iteration comparison
# --------------------------------------------------------------------------

def _build_contact_sheet(candidates: list, out_path: Path) -> Optional[Path]:
    """Stack every iteration's preview into one labeled image for eyeballing.

    There is no reference to score against, so the loop cannot rank iterations —
    this exists so a human can compare them at a glance and see whether a later
    critique round actually improved the asset or made it worse.
    """
    if Image is None or len(candidates) < 2:
        return None

    tile_w = 640
    label_h = 26
    tiles = []
    for c in candidates:
        img = Image.open(c["render"]).convert("RGB")
        h = max(1, round(img.height * tile_w / img.width))
        tiles.append((c, img.resize((tile_w, h), Image.LANCZOS)))

    total_h = sum(label_h + t.height for _, t in tiles)
    sheet = Image.new("RGB", (tile_w, total_h), color=(24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    y = 0
    for c, tile in tiles:
        n = c.get("n_diffs")
        note = "critic: n/a" if n is None else f"critic found {n} diff(s)"
        draw.text((6, y + 7), f"iter {c['iter']}   —   {note}", fill=(255, 255, 255))
        y += label_h
        sheet.paste(tile, (0, y))
        y += tile.height

    sheet.save(out_path)
    return out_path


def _print_comparison(candidates: list, default_iter: int, contact_sheet: Optional[Path]) -> None:
    """Tell the user what rendered and that the default is not necessarily best."""
    print(f"\n[choose] {len(candidates)} iteration(s) rendered successfully.", flush=True)
    if len(candidates) > 1:
        print(
            "         NOTE: a later iteration is not necessarily better — a critique round "
            "can make\n         the asset worse. There is no reference to score against, so "
            "compare these yourself:",
            flush=True,
        )
    for c in candidates:
        n = c.get("n_diffs")
        note = "critic: n/a" if n is None else f"critic found {n} diff(s)"
        mark = " <- default" if c["iter"] == default_iter else ""
        print(f"           [{c['iter']}] {note:24s} {c['render']}{mark}", flush=True)
    if contact_sheet is not None:
        print(f"         side-by-side comparison: {contact_sheet}", flush=True)
    if len(candidates) > 1:
        print(
            f"         (diff counts are the critic's opinion, not a score — eyeball the "
            f"previews.\n          re-run with --pick to choose which iteration to export.)",
            flush=True,
        )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def generate(
    out_glb: Path,
    prompt: Optional[str] = None,
    style_image: Optional[Path] = None,
    iters: int = 3,
    coder: str = DEFAULT_CODER,
    critic: str = DEFAULT_CRITIC,
    director: str = DEFAULT_DIRECTOR,
    choose: Optional[Callable[[list, Optional[Path]], int]] = None,
) -> dict:
    """Generate a stylized low-poly GLB asset from text and/or a style image.

    Only a script that actually rendered is ever exported. Every successful
    iteration is kept as a candidate, since "rendered" means "didn't crash", NOT
    "is better" — a critique round can make the asset worse. With no reference to
    score against, the loop cannot rank them, so:
      - by default the most recent successful render is exported, and every
        candidate is listed (plus a contact sheet) for the user to eyeball;
      - pass `choose(candidates, contact_sheet) -> iter` to select one instead
        (the CLI's --pick supplies an interactive chooser).

    Returns a dict for the web UI:
        {
          "glb":           Path to the exported .glb,
          "script":        the exported script,
          "spec":          the art director's build spec,
          "log":           [{"iter", "render", "script", "traceback", "critique"}, ...],
          "best_iter":     index of the iteration whose script was exported,
          "candidates":    [{"iter", "script", "script_path", "render", "n_diffs"}, ...],
          "contact_sheet": Path to the labeled iteration comparison, or None,
        }

    Raises ValueError on bad inputs (including coder == critic), and RuntimeError
    if no iteration ever rendered successfully or the final export fails.
    """
    if prompt is None and style_image is None:
        raise ValueError("generate() requires a prompt, a style_image, or both")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")

    # A model critiquing its own output rubber-stamps it. This is load-bearing:
    # without an independent critic the loop stops improving after iteration 1.
    if coder == critic:
        raise ValueError(
            f"coder and critic must be different models (both are {coder!r}); "
            "a model critiquing its own output rubber-stamps it"
        )

    out_glb = Path(out_glb)
    out_py = out_glb.with_suffix(".py")
    style_image = Path(style_image) if style_image is not None else None
    workdir = Path(tempfile.mkdtemp(prefix="generate_"))

    # --- Art direction -----------------------------------------------------
    if prompt and style_image:
        brief = f"{prompt!r} + style image {style_image.name}"
    elif style_image:
        brief = f"style image {style_image.name}"
    else:
        brief = repr(prompt)
    print(f"[spec] Art-directing {brief} with {director} ...", flush=True)

    spec, _ = gen_spec_from_prompt(prompt, director, style_image=style_image)
    parts = spec.get("parts", [])
    pieces = sum(p.get("count", 1) for p in parts)
    palette = ", ".join(c.get("hex", "?") for c in spec.get("palette", []))
    print(
        f"       object={spec.get('object')!r} | {len(parts)} part type(s) / {pieces} piece(s) "
        f"| palette: {palette or '(none)'} | target: {spec.get('polycount_target', 'n/a')}",
        flush=True,
    )

    # --- First build -------------------------------------------------------
    print(f"[build] Writing procedural asset script with {coder} ...", flush=True)
    script, _ = gen_script(spec, None, None, None, coder, mode="asset")

    # --- Feedback loop -----------------------------------------------------
    # best_script is the most recent script that RENDERED without a traceback,
    # i.e. the last known-good code. Every revision the coder makes is
    # unvalidated until the next render proves it — and the loop's final
    # revision is never rendered at all. Exporting `script` would therefore risk
    # shipping code that crashes, so the export uses best_script instead.
    # Every iteration whose script rendered cleanly becomes a candidate for
    # export. "Rendered" only means "didn't crash" — it does NOT mean "better",
    # so all candidates are kept for comparison rather than assuming the last
    # one wins.
    candidates: list = []
    last_traceback: Optional[str] = None
    log: list = []

    for k in range(iters):
        script_path = workdir / f"script_{k}.py"
        script_path.write_text(script, encoding="utf-8")

        print(f"[iter {k}] rendering preview ...", flush=True)
        # RGB grid only; render_rig also produces a silhouette, unused here.
        grid, _sil, err = render(script_path, workdir / f"preview_{k}")

        if err:
            # Send the verbatim traceback straight back to the coder and retry.
            # A crashing script is never a candidate.
            print(f"[iter {k}] preview render failed; sending traceback back to {coder}", flush=True)
            last_traceback = err
            log.append({"iter": k, "render": None, "script": None, "traceback": err, "critique": None})
            script, _ = gen_script(spec, script, [{"error": err}], None, coder, mode="asset")
            continue

        # This script rendered cleanly, so it is known-good. Capture it BEFORE
        # the critique below regenerates `script` into something unvalidated.
        candidate = {
            "iter": k,
            "script": script,
            "script_path": str(script_path),
            "render": str(grid),
            "n_diffs": None,
        }
        candidates.append(candidate)

        print(f"[iter {k}] {critic} critiquing against the spec ...", flush=True)
        diffs, _ = gen_asset_critique(grid, spec, critic, style_image=style_image)
        candidate["n_diffs"] = len(diffs)
        log.append({
            "iter": k, "render": str(grid), "script": str(script_path),
            "traceback": None, "critique": diffs,
        })

        if not diffs:
            print(f"[iter {k}] critic reports no diffs — asset realizes the spec, stopping", flush=True)
            break

        top = ", ".join(f"{d.get('part')}: {d.get('issue', '')[:40]}" for d in diffs[:3])
        print(f"[iter {k}] {len(diffs)} diff(s) -> {coder} revising | {top}", flush=True)
        script, _ = gen_script(spec, script, diffs, grid, coder, mode="asset")

    # --- Choose what to export ---------------------------------------------
    # Nothing ever rendered: a real total failure, worth surfacing loudly.
    if not candidates:
        raise RuntimeError(
            f"No iteration rendered successfully in {iters} attempt(s) — nothing "
            f"safe to export. Last traceback:\n{last_traceback}"
        )

    contact_sheet = _build_contact_sheet(candidates, out_glb.with_name(f"{out_glb.stem}_iterations.png"))

    # Default is the most recent successful render — never the loop's final
    # unvalidated revision. It is the default, not a claim that it is best.
    default_iter = candidates[-1]["iter"]
    _print_comparison(candidates, default_iter, contact_sheet)

    chosen_iter = default_iter
    if choose is not None and len(candidates) > 1:
        picked = choose(candidates, contact_sheet)
        if not any(c["iter"] == picked for c in candidates):
            raise ValueError(f"choose() returned iter {picked}, which is not a candidate")
        chosen_iter = picked

    chosen = next(c for c in candidates if c["iter"] == chosen_iter)
    chosen_script = chosen["script"]

    if chosen_script is not script:
        print(
            f"[export] the loop's final revision was never validated by a render; "
            f"exporting the known-good script from iter {chosen_iter} instead",
            flush=True,
        )

    # --- Export ------------------------------------------------------------
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(chosen_script, encoding="utf-8")

    print(f"[export] Exporting GLB (with materials) from iter {chosen_iter}'s script ...", flush=True)
    final_script_path = workdir / "final_script.py"
    final_script_path.write_text(chosen_script, encoding="utf-8")
    glb_path, err = export_glb(final_script_path, out_glb)
    if err:
        raise RuntimeError(f"GLB export failed:\n{err}")

    print(f"Wrote {glb_path}", flush=True)
    print(f"Wrote {out_py}", flush=True)
    if contact_sheet is not None:
        print(f"Wrote {contact_sheet}  (compare iterations here)", flush=True)

    return {
        "glb": glb_path,
        "script": chosen_script,
        "spec": spec,
        "log": log,
        "best_iter": chosen_iter,
        "candidates": candidates,
        "contact_sheet": contact_sheet,
    }


def _interactive_chooser(candidates: list, contact_sheet: Optional[Path]) -> int:
    """Ask the user which iteration to export. Used by --pick."""
    default = candidates[-1]["iter"]
    valid = [c["iter"] for c in candidates]
    if contact_sheet is not None:
        print(f"\nOpen {contact_sheet} to compare the iterations side by side.", flush=True)
    while True:
        try:
            raw = input(f"Which iteration should be exported? {valid} [default {default}]: ").strip()
        except EOFError:
            # Non-interactive stdin (piped/CI): fall back to the default.
            print(f"(no input available — using default iter {default})", flush=True)
            return default
        if not raw:
            return default
        try:
            picked = int(raw)
        except ValueError:
            print(f"  Not a number. Choose one of {valid}.", flush=True)
            continue
        if picked in valid:
            return picked
        print(f"  iter {picked} did not render successfully. Choose one of {valid}.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", type=str, default=None, help="Text description of the asset.")
    parser.add_argument("--style-image", type=Path, default=None, help="Image used as STYLE INSPIRATION (not a reconstruction target).")
    parser.add_argument("--out", type=Path, required=True, help="Output .glb path.")
    parser.add_argument("--iters", type=int, default=3, help="Max render/critique/revise iterations.")
    parser.add_argument("--coder", type=str, default=DEFAULT_CODER, help="Model that writes the Blender script.")
    parser.add_argument("--critic", type=str, default=DEFAULT_CRITIC, help="Model that critiques renders (must differ from --coder).")
    parser.add_argument("--director", type=str, default=DEFAULT_DIRECTOR, help="Model that writes the build spec.")
    parser.add_argument("--log-json", type=Path, default=None, help="Optional path to write the run log as JSON.")
    parser.add_argument("--pick", action="store_true", help="After the loop, compare iteration previews and choose which one to export.")
    args = parser.parse_args()

    if args.prompt is None and args.style_image is None:
        parser.error("one of --prompt or --style-image is required (both may be given)")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.style_image is not None and not args.style_image.is_file():
        parser.error(f"--style-image does not exist: {args.style_image}")
    if args.iters < 1:
        parser.error("--iters must be >= 1")
    if args.coder == args.critic:
        parser.error("--coder and --critic must be different models (a model critiquing its own output rubber-stamps it)")

    try:
        result = generate(
            args.out,
            prompt=args.prompt,
            style_image=args.style_image,
            iters=args.iters,
            coder=args.coder,
            critic=args.critic,
            director=args.director,
            choose=_interactive_chooser if args.pick else None,
        )
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.log_json:
        payload = {
            "glb": str(result["glb"]),
            "spec": result["spec"],
            "log": result["log"],
            "best_iter": result["best_iter"],
            "contact_sheet": str(result["contact_sheet"]) if result["contact_sheet"] else None,
        }
        args.log_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.log_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
