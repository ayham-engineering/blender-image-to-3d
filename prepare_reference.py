"""Prepare a single reference image as a 6-view comparison grid.

Two grids, both mirroring the 768x512 (3x2 tiles of 256x256) layout produced
by render_rig.render():

  prepare_reference(image_path, out_dir)
      -> reference_grid.png: the RGB photograph tiled into all 6 cells.

  prepare_reference_silhouette(image_path, out_dir)
      -> reference_silhouette_grid.png: a black-on-white segmentation mask of
         the object, tiled into all 6 cells, aligned with the RGB grid.

The silhouette grid is what the primary objective (silhouette IoU) compares
against, so a bad mask silently destroys the whole signal — hence the
degeneracy warning.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from render_rig import GRID_SIZE, GRID_COLS, TILE_SIZE, VIEW_ORDER


def _fit_to_tile_geometry(width: int, height: int):
    """Return (new_w, new_h, off_x, off_y) to fit a WxH image inside one tile,
    preserving aspect ratio and centering. Shared by the RGB and silhouette
    paths so the mask lands exactly where the photograph does.
    """
    scale = min(TILE_SIZE / width, TILE_SIZE / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    off_x = (TILE_SIZE - new_w) // 2
    off_y = (TILE_SIZE - new_h) // 2
    return new_w, new_h, off_x, off_y


def _tile_into_grid(tile: "Image.Image", bg_color) -> "Image.Image":
    """Tile a single TILE_SIZE image into all 6 grid cells."""
    grid = Image.new("RGB", GRID_SIZE, color=bg_color)
    for idx in range(len(VIEW_ORDER)):
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        grid.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
    return grid


def prepare_reference(image_path: Path, out_dir: Path) -> Path:
    """Tile a single reference image into the same 6-view 768x512 grid layout.

    NOTE / known limitation: since only one reference view is available, the
    same image is tiled into all 6 cells. This weakens the multi-view signal
    compared to a true 6-view reference set, since it cannot capture
    view-dependent geometry differences (e.g. an object correct from the
    front but wrong from behind will score as if it were correct everywhere).
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to build the reference grid: pip install Pillow")

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(image_path).convert("RGB")
    if source.size != (TILE_SIZE, TILE_SIZE):
        # Scale to fit within TILE_SIZE preserving aspect ratio, then pad
        # with the same flat mid-gray as the render background, rather than
        # stretching non-square images (which would distort proportions and
        # make geometry look wrong that isn't).
        new_w, new_h, off_x, off_y = _fit_to_tile_geometry(source.width, source.height)
        resized = source.resize((new_w, new_h), Image.LANCZOS)
        padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE), color=(128, 128, 128))
        padded.paste(resized, (off_x, off_y))
        source = padded

    grid = _tile_into_grid(source, bg_color=(128, 128, 128))
    grid_path = out_dir / "reference_grid.png"
    grid.save(grid_path)
    return grid_path


# --------------------------------------------------------------------------
# Silhouette extraction
# --------------------------------------------------------------------------

def clean_mask(mask):
    """Post-process a single-view boolean foreground mask.

    1. Keep only the largest connected component, dropping stray specks that
       segmentation leaves in the background (e.g. rembg flecks in grass).
    2. Fill enclosed holes, so a hollow woven frame (seat/back that is mostly
       gaps) becomes a solid silhouette. This also matches the render side
       better: Blender renders solid geometry, not cord gaps.

    Operates on ONE view's mask. It must not be applied across a tiled 6-view
    grid, or the largest-component step would collapse the six tiles into one.
    Shared verbatim with score.py so the reference and render silhouettes go
    through identical post-processing.
    """
    import numpy as np
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask

    labeled, n = ndimage.label(mask)
    if n > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # ignore the background label
        mask = labeled == counts.argmax()

    return ndimage.binary_fill_holes(mask)


def _segment_foreground(source: "Image.Image"):
    """Return (bool_mask, method_name) with True = foreground object pixel.

    Prefers rembg (a learned matting model) if it imports cleanly; otherwise
    falls back to an Otsu threshold with border-based polarity detection.
    """
    import numpy as np

    # Preferred path: rembg.
    try:
        from rembg import remove
    except Exception:
        remove = None

    if remove is not None:
        try:
            cut = remove(source)  # RGBA with matting in the alpha channel
            alpha = np.array(cut.convert("RGBA"))[:, :, 3]
            return alpha > 128, "rembg"
        except Exception as e:
            print(f"[prepare_reference_silhouette] rembg failed ({e}); falling back to Otsu.")

    # Fallback path: Otsu threshold.
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu

    gray = rgb2gray(np.array(source))  # H x W, floats in [0, 1]
    try:
        thresh = threshold_otsu(gray)
    except ValueError:
        # Degenerate (single-value) image: no foreground can be found.
        return np.zeros(gray.shape, dtype=bool), "otsu"
    above = gray > thresh

    # Polarity: the object is whichever class is LESS common on the border,
    # since the frame edge is almost always background.
    border = np.concatenate([above[0, :], above[-1, :], above[:, 0], above[:, -1]])
    mask = ~above if border.mean() > 0.5 else above
    return mask, "otsu"


def prepare_reference_silhouette(image_path: Path, out_dir: Path) -> Path:
    """Segment the object from its background and tile a black-on-white mask
    into the same 6-view 768x512 grid layout as the render silhouette grid.

    Warns if the mask looks degenerate (>90% or <5% foreground), which usually
    means segmentation failed and the IoU signal would be meaningless.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to build the reference silhouette: pip install Pillow")

    import numpy as np

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(image_path).convert("RGB")
    mask, method = _segment_foreground(source)

    # Fill holes + keep the largest component at full segmentation resolution,
    # BEFORE the degeneracy check below (so the check reflects the final mask)
    # and before downscaling. This rescues hollow woven objects whose interior
    # would otherwise read as background and collapse the foreground fraction.
    mask = clean_mask(mask)

    # Build one black-on-white tile, fitting the mask into the tile with the
    # exact same geometry the RGB reference uses so the two grids align.
    new_w, new_h, off_x, off_y = _fit_to_tile_geometry(source.width, source.height)
    mask_img = Image.fromarray((mask.astype("uint8") * 255), mode="L").resize(
        (new_w, new_h), Image.NEAREST
    )
    tile = Image.new("L", (TILE_SIZE, TILE_SIZE), color=255)  # white background
    black = Image.new("L", (new_w, new_h), color=0)           # black object
    tile.paste(black, (off_x, off_y), mask=mask_img)          # paste black where mask is set
    tile_rgb = tile.convert("RGB")

    # Degeneracy check on the composed tile (foreground = black pixels), run
    # AFTER fill+cleanup — so if it still warns, hole-filling didn't rescue the
    # mask and the segmentation genuinely failed.
    tile_arr = np.array(tile)
    fg_fraction = float((tile_arr == 0).mean())
    if fg_fraction > 0.90 or fg_fraction < 0.05:
        print(
            f"[prepare_reference_silhouette] WARNING: mask still degenerate after "
            f"fill+cleanup ({fg_fraction:.1%} foreground via {method}). Segmentation "
            f"likely failed; the silhouette IoU signal will be unreliable. Eyeball "
            f"the saved mask before trusting the scores."
        )
    else:
        print(f"[prepare_reference_silhouette] mask OK: {fg_fraction:.1%} foreground via {method} (after fill+cleanup).")

    grid = _tile_into_grid(tile_rgb, bg_color=(255, 255, 255))
    grid_path = out_dir / "reference_silhouette_grid.png"
    grid.save(grid_path)
    return grid_path


if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) < 2:
        print("Usage: python prepare_reference.py <image_path> [out_dir]", file=sys.stderr)
        sys.exit(1)

    image_arg = Path(sys.argv[1])
    out_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "prepare_reference_output"

    grid_path = prepare_reference(image_arg, out_dir_arg)
    print(f"Reference grid written to:            {grid_path}")
    sil_path = prepare_reference_silhouette(image_arg, out_dir_arg)
    print(f"Reference silhouette grid written to: {sil_path}")
