"""generate.py — turn a reference image into a 3D model (.glb).

Standalone pipeline (no benchmark / loop / score machinery):

  1. gen_spec(image, model)            -> structured part spec
  2. gen_script(spec, ..., model)      -> Blender geometry-only Python script
  3. Run that script in Blender headless and EXPORT to GLB (no rendering)
  4. Save the .glb and the generated .py next to it

Optional refinement (--iters N, default 1): for N > 1, render a preview of the
current model, hand it back to the coder alongside the reference with a plain
"make it closer" instruction, regenerate, and re-export. No scoring — just
image-in, better-script-out, up to N generations total.

Usage:
    python generate.py --image chair.jpg --out chair.glb
    python generate.py --image chair.jpg --out chair.glb --iters 3

Reuses render_rig's Blender-binary resolution, traceback scanning, and the
geometry-only wrapper approach (clear scene, exec geometry script). The final
Blender step exports GLB instead of rendering six views.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from render_rig import find_blender_binary, _extract_traceback, render
from models import gen_spec, gen_script

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# The task pins Claude Opus for both spec and coder; overridable via --model.
DEFAULT_MODEL = "claude-opus-4-8"


# --------------------------------------------------------------------------
# Blender wrapper: clear scene, exec geometry script, export GLB
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
    # modifiers, and use +Y up (Three.js / glTF convention).
    bpy.ops.export_scene.gltf(
        filepath=GLB_OUT_PATH,
        export_format="GLB",
        export_apply=True,   # apply modifiers on export
        export_yup=True,     # +Y up
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
    """Run the geometry script in Blender headless and export it to GLB.

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
# Refinement helpers
# --------------------------------------------------------------------------

def _compose_reference_and_preview(ref_image: Path, preview_grid: Path, out_path: Path) -> Path:
    """Build one side-by-side image: reference (left) | preview render (right).

    gen_script accepts a single image, so we hand the coder both at once. The
    accompanying text instruction names which side is which.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for the refinement pass: pip install Pillow")

    target_h = 512
    gap = 16

    ref = Image.open(ref_image).convert("RGB")
    prev = Image.open(preview_grid).convert("RGB")

    def _scale_to_h(img):
        if img.height == target_h:
            return img
        new_w = max(1, round(img.width * target_h / img.height))
        return img.resize((new_w, target_h), Image.LANCZOS)

    ref = _scale_to_h(ref)
    prev = _scale_to_h(prev)

    canvas = Image.new("RGB", (ref.width + gap + prev.width, target_h), color=(255, 255, 255))
    canvas.paste(ref, (0, 0))
    canvas.paste(prev, (ref.width + gap, 0))
    canvas.save(out_path)
    return out_path


_REFINE_INSTRUCTION = (
    "The attached image places the REFERENCE (left) next to a PREVIEW render of your "
    "current model (right, shown as a 6-view grid). Make the geometry match the reference "
    "more closely — improve the shapes, proportions, and placement of parts. Return the "
    "complete corrected script."
)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def generate(image: Path, out_glb: Path, iters: int = 1, model: str = DEFAULT_MODEL) -> Path:
    """Turn a reference image into a GLB model. Returns the GLB path.

    Raises RuntimeError with the Blender traceback if the final export fails.
    """
    image = Path(image)
    out_glb = Path(out_glb)
    out_py = out_glb.with_suffix(".py")
    workdir = Path(tempfile.mkdtemp(prefix="generate_"))

    print(f"[1/2] Analyzing reference with {model} ...", flush=True)
    spec, _ = gen_spec(image, model)
    print(f"      spec: object={spec.get('object')!r}, {len(spec.get('parts', []))} part(s)", flush=True)

    print(f"[2/2] Writing Blender geometry script with {model} ...", flush=True)
    script, _ = gen_script(spec, None, None, None, model)

    # Optional refinement: render a preview, hand it back with the reference.
    for refine_i in range(1, iters):
        print(f"[refine {refine_i}/{iters - 1}] rendering preview ...", flush=True)
        script_path = workdir / f"script_iter{refine_i}.py"
        script_path.write_text(script, encoding="utf-8")
        grid, _sil, err = render(script_path, workdir / f"preview_{refine_i}")

        if err:
            # No preview available (the script crashed). Send the verbatim
            # traceback back so the coder can fix it, then continue refining.
            print(f"[refine {refine_i}/{iters - 1}] preview render failed; sending traceback back to coder", flush=True)
            script, _ = gen_script(spec, script, [{"error": err}], None, model)
            continue

        composite = _compose_reference_and_preview(image, grid, workdir / f"composite_{refine_i}.png")
        print(f"[refine {refine_i}/{iters - 1}] asking {model} to make it closer ...", flush=True)
        script, _ = gen_script(spec, script, [{"instruction": _REFINE_INSTRUCTION}], composite, model)

    # Save the final generated script next to the GLB.
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(script, encoding="utf-8")

    print("Exporting to GLB via Blender headless ...", flush=True)
    final_script_path = workdir / "final_script.py"
    final_script_path.write_text(script, encoding="utf-8")
    glb_path, err = export_glb(final_script_path, out_glb)
    if err:
        raise RuntimeError(f"GLB export failed:\n{err}")

    print(f"Wrote {glb_path}", flush=True)
    print(f"Wrote {out_py}", flush=True)
    return glb_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="Reference image to reconstruct.")
    parser.add_argument("--out", type=Path, required=True, help="Output .glb path.")
    parser.add_argument("--iters", type=int, default=1, help="Total generations; >1 enables refinement passes.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model for spec + coder.")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"--image does not exist: {args.image}")
    if args.iters < 1:
        parser.error("--iters must be >= 1")

    try:
        generate(args.image, args.out, iters=args.iters, model=args.model)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
