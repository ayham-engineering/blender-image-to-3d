"""Shape-first scoring for the render benchmark.

PRIMARY objective: silhouette IoU loss (1 - IoU) on binary black-on-white
masks. IoU measures ONLY shape overlap — it ignores color, texture, and
lighting entirely, which are exactly the things the geometry coder cannot
control. Lower is better (a perfect shape match is loss 0).

LPIPS, SSIM, and MSE (computed on the RGB grids) are kept in the metrics dict
for logging/diagnostics only. They are NOT the objective: LPIPS on RGB
compares a textured photograph against untextured gray renders, so its
distance is dominated by appearance the coder can't fix — which is why the
loop used to plateau near the score of unrelated images.

All scores are computed on CPU for determinism across machines/GPUs.
The LPIPS model (net='alex') is loaded once at module level.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim

from render_rig import GRID_COLS, GRID_ROWS
from prepare_reference import clean_mask

try:
    import lpips
except ImportError:  # pragma: no cover
    raise ImportError(
        "lpips is required. Install with: pip install lpips"
    )


# Determinism: set seeds and disable nondeterministic CUDA ops
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Load LPIPS model once at module level (not per call)
_lpips_model = lpips.LPIPS(net="alex", verbose=False).to("cpu")
_lpips_model.eval()

# Silhouette masks are black object (0) on white background (255).
_FOREGROUND_THRESHOLD = 128


def _load_and_normalize_image(image_path: Path) -> torch.Tensor:
    """Load a PNG, normalize from [0, 255] to [-1, 1], return as (H, W, C) tensor on CPU."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    # [0, 255] -> [0, 1] -> [-1, 1]
    img_normalized = 2.0 * (img_array / 255.0) - 1.0
    return torch.from_numpy(img_normalized).to("cpu")


def _rgb_metrics(ref_grid_path: Path, render_grid_path: Path) -> dict:
    """LPIPS/SSIM/MSE on the RGB grids — diagnostics only, never the objective."""
    ref_tensor = _load_and_normalize_image(Path(ref_grid_path))       # (H, W, C) in [-1, 1]
    render_tensor = _load_and_normalize_image(Path(render_grid_path))

    # Resize render to match ref if dimensions differ.
    if render_tensor.shape[:2] != ref_tensor.shape[:2]:
        h_ref, w_ref = ref_tensor.shape[:2]
        render_np = ((render_tensor.numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
        render_pil = Image.fromarray(render_np).resize((w_ref, h_ref), Image.LANCZOS)
        render_np = np.array(render_pil, dtype=np.float32)
        render_tensor = torch.from_numpy(2.0 * (render_np / 255.0) - 1.0).to("cpu")

    ref_batch = ref_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    render_batch = render_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        lpips_value = float(_lpips_model(ref_batch, render_batch).item())

    ref_np = ((ref_tensor.numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
    render_np = ((render_tensor.numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
    ssim_value = float(skimage_ssim(ref_np, render_np, channel_axis=2, data_range=255))
    mse_value = float(np.mean((ref_np.astype(np.float32) - render_np.astype(np.float32)) ** 2))

    return {"lpips": lpips_value, "ssim": ssim_value, "mse": mse_value}


def _clean_grid_mask(mask: np.ndarray) -> np.ndarray:
    """Apply clean_mask (fill holes + largest component) to each view tile.

    The silhouette grids are GRID_ROWS x GRID_COLS tiles, each a different view.
    Cleaning must be per-tile: applied to the whole grid, the largest-component
    step would keep only one tile and blank the other five.
    """
    h, w = mask.shape
    th, tw = h // GRID_ROWS, w // GRID_COLS
    out = mask.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ys, xs = slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw)
            out[ys, xs] = clean_mask(mask[ys, xs])
    return out


def _foreground_mask(sil_path: Path, shape=None) -> np.ndarray:
    """Load a black-on-white silhouette PNG and return a bool foreground mask.

    Foreground (the object) is the dark pixels. If `shape` (H, W) is given and
    differs, the mask is nearest-resized to it so ref/render masks align.

    Both the reference and render silhouettes are passed through the identical
    per-view clean_mask post-processing here, so IoU compares solid shape to
    solid shape by construction, not by luck.
    """
    img = Image.open(sil_path).convert("L")
    if shape is not None and (img.height, img.width) != shape:
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
    arr = np.array(img)
    mask = arr < _FOREGROUND_THRESHOLD
    return _clean_grid_mask(mask)


def _silhouette_iou(ref_sil_path: Path, render_sil_path: Path) -> float:
    """Intersection-over-union of two binary silhouette masks."""
    ref_mask = _foreground_mask(ref_sil_path)
    render_mask = _foreground_mask(render_sil_path, shape=ref_mask.shape)

    intersection = np.logical_and(ref_mask, render_mask).sum()
    union = np.logical_or(ref_mask, render_mask).sum()
    if union == 0:
        # Neither mask has any foreground: nothing to disagree on -> perfect.
        return 1.0
    return float(intersection) / float(union)


def score(ref_grid_path: Path, render_grid_path: Path, ref_sil_path: Path, render_sil_path: Path) -> tuple[float, dict]:
    """Score a render against a reference. Primary objective is silhouette IoU loss.

    Args:
        ref_grid_path:    reference 768x512 RGB PNG.
        render_grid_path: rendered 768x512 RGB PNG.
        ref_sil_path:     reference 768x512 black-on-white silhouette PNG.
        render_sil_path:  rendered 768x512 black-on-white silhouette PNG.

    Returns:
        (iou_loss, metrics) where iou_loss = 1 - IoU (lower is better) and
        metrics = {"iou_loss", "iou", "lpips", "ssim", "mse"}. LPIPS/SSIM/MSE
        are logging-only diagnostics, not the objective.
    """
    iou = _silhouette_iou(Path(ref_sil_path), Path(render_sil_path))
    iou_loss = 1.0 - iou

    metrics = {"iou_loss": iou_loss, "iou": iou}
    metrics.update(_rgb_metrics(ref_grid_path, render_grid_path))
    return iou_loss, metrics


if __name__ == "__main__":
    tmp = Path(tempfile.gettempdir())

    # Build a reference silhouette: black square on white.
    ref_sil = Image.new("L", (768, 512), color=255)
    ref_arr = np.array(ref_sil)
    ref_arr[150:350, 250:500] = 0
    Image.fromarray(ref_arr).convert("RGB").save(tmp / "score_ref_sil.png")

    rgb = Image.new("RGB", (768, 512), color=(128, 128, 128))
    rgb.save(tmp / "score_rgb.png")

    # Test 1: identical silhouette -> IoU 1.0, loss 0.0.
    loss, m = score(tmp / "score_rgb.png", tmp / "score_rgb.png",
                    tmp / "score_ref_sil.png", tmp / "score_ref_sil.png")
    print(f"Identical silhouette: iou_loss={loss:.4f} iou={m['iou']:.4f} "
          f"(lpips={m['lpips']:.4f} ssim={m['ssim']:.4f} mse={m['mse']:.1f})")
    assert abs(loss) < 1e-6, f"expected 0 loss for identical masks, got {loss}"

    # Test 2: empty render silhouette (all white) -> IoU 0, loss 1.0 (the floor).
    empty_sil = Image.new("RGB", (768, 512), color=(255, 255, 255))
    empty_sil.save(tmp / "score_empty_sil.png")
    loss2, m2 = score(tmp / "score_rgb.png", tmp / "score_rgb.png",
                     tmp / "score_ref_sil.png", tmp / "score_empty_sil.png")
    print(f"Empty render silhouette (floor): iou_loss={loss2:.4f} iou={m2['iou']:.4f}")
    assert abs(loss2 - 1.0) < 1e-6, f"expected loss 1.0 for empty render, got {loss2}"

    # Test 3: partially overlapping square -> intermediate loss.
    shifted = np.full((512, 768), 255, dtype=np.uint8)
    shifted[200:400, 300:550] = 0  # shifted + resized overlap with ref
    Image.fromarray(shifted).convert("RGB").save(tmp / "score_shifted_sil.png")
    loss3, m3 = score(tmp / "score_rgb.png", tmp / "score_rgb.png",
                     tmp / "score_ref_sil.png", tmp / "score_shifted_sil.png")
    print(f"Partial overlap: iou_loss={loss3:.4f} iou={m3['iou']:.4f}")
    assert 0.0 < loss3 < 1.0, f"expected intermediate loss, got {loss3}"

    print("\nAll silhouette-IoU score assertions passed.")
