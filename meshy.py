"""meshy.py — a Meshy API client for text-to-3D and image-to-3D generation.

Base URL: https://api.meshy.ai   Auth: Bearer <MESHY_API_KEY>

Two entry points, both returning {"glb_path", "task_ids", "credits_used", ...}:

    text_to_3d(prompt, ...)          POST openapi/v2/text-to-3d
        Two-step: mode="preview" -> poll -> mode="refine" (textures) -> poll.

    image_to_3d(image_url_or_path, ...)   POST openapi/v1/image-to-3d
        Single-step: no preview/refine loop.

Endpoint versions differ between the two APIs (v2 for text, v1 for image) —
that is per the live docs, not a typo:
    https://docs.meshy.ai/en/api/text-to-3d
    https://docs.meshy.ai/en/api/image-to-3d
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Callable, Optional

import requests


BASE_URL = "https://api.meshy.ai"

TEXT_TO_3D_PATH = "/openapi/v2/text-to-3d"
IMAGE_TO_3D_PATH = "/openapi/v1/image-to-3d"

# Terminal task states.
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_CANCELED = "CANCELED"
TERMINAL_STATUSES = {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELED}

# Total wall-clock budget for a whole call, across every poll phase.
MAX_TOTAL_WAIT_SECONDS = 5 * 60

POLL_INITIAL_INTERVAL = 2.0
POLL_MAX_INTERVAL = 15.0
POLL_BACKOFF_FACTOR = 1.5

REQUEST_TIMEOUT = 60


class MeshyError(RuntimeError):
    """Any Meshy API failure: auth, task FAILED/CANCELED, or timeout."""


def _api_key() -> str:
    """Read the API key from the environment.

    Prefers MESHY_API_KEY; falls back to the misspelled MESHI_API_KEY (which
    exists in some local .env files) so a typo doesn't silently break the app.
    """
    key = os.environ.get("MESHY_API_KEY")
    if key:
        return key
    typo = os.environ.get("MESHI_API_KEY")
    if typo:
        print("[meshy] WARNING: using MESHI_API_KEY — the correct name is MESHY_API_KEY.")
        return typo
    raise MeshyError(
        "MESHY_API_KEY is not set. Export it (or add it to .env) before generating: "
        "export MESHY_API_KEY=msy_..."
    )


def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


def _log(on_progress: Optional[Callable[[str], None]], message: str) -> None:
    print(f"[meshy] {message}", flush=True)
    if on_progress is not None:
        on_progress(message)


def _post_task(path: str, payload: dict) -> str:
    """POST a task-creation request and return the new task id."""
    url = BASE_URL + path
    resp = requests.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 401:
        raise MeshyError("Meshy rejected the API key (401). Check MESHY_API_KEY.")
    if resp.status_code == 402:
        raise MeshyError(f"Meshy reports insufficient credits (402): {resp.text}")
    if not resp.ok:
        raise MeshyError(f"POST {path} failed [{resp.status_code}]: {resp.text}")

    data = resp.json()
    # Creation responses look like {"result": "<id>"}; accept {"id": ...} too.
    task_id = data.get("result") or data.get("id")
    if not task_id:
        raise MeshyError(f"POST {path} returned no task id: {data}")
    return task_id


def _get_task(path: str, task_id: str) -> dict:
    url = f"{BASE_URL}{path}/{task_id}"
    resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
    if not resp.ok:
        raise MeshyError(f"GET {path}/{task_id} failed [{resp.status_code}]: {resp.text}")
    return resp.json()


def _poll_until_done(
    path: str,
    task_id: str,
    deadline: float,
    label: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Poll a task with exponential backoff until it reaches a terminal state.

    `deadline` is an absolute time.monotonic() value shared across all phases of
    a call, so the 5-minute cap covers preview + refine together rather than
    resetting per phase.
    """
    interval = POLL_INITIAL_INTERVAL
    last_progress = -1

    while True:
        task = _get_task(path, task_id)
        status = task.get("status")
        progress = task.get("progress", 0)

        if progress != last_progress:
            _log(on_progress, f"{label}: {status} {progress}%")
            last_progress = progress

        if status == STATUS_SUCCEEDED:
            return task
        if status == STATUS_FAILED:
            err = (task.get("task_error") or {}).get("message") or "(no error message)"
            raise MeshyError(f"{label} FAILED: {err}")
        if status == STATUS_CANCELED:
            raise MeshyError(f"{label} was CANCELED.")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MeshyError(
                f"{label} timed out after {MAX_TOTAL_WAIT_SECONDS}s "
                f"(last status {status} at {progress}%). Task id: {task_id}"
            )

        time.sleep(min(interval, remaining))
        interval = min(interval * POLL_BACKOFF_FACTOR, POLL_MAX_INTERVAL)


def _download(url: str, out_path: Path, on_progress: Optional[Callable[[str], None]] = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(on_progress, f"downloading GLB -> {out_path.name}")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        if not resp.ok:
            raise MeshyError(f"Downloading the GLB failed [{resp.status_code}]: {url}")
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    return out_path


def _glb_url(task: dict) -> str:
    url = (task.get("model_urls") or {}).get("glb")
    if not url:
        raise MeshyError(f"Task succeeded but returned no GLB url. model_urls={task.get('model_urls')}")
    return url


def _as_image_url(image_url_or_path) -> str:
    """Pass through an http(s) URL or data URI; encode a local file as a data URI."""
    s = str(image_url_or_path)
    if s.startswith(("http://", "https://", "data:")):
        return s

    path = Path(s)
    if not path.is_file():
        raise MeshyError(f"Image not found and not a URL: {s}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if mime not in ("image/jpeg", "image/png"):
        raise MeshyError(f"Meshy accepts .jpg/.jpeg/.png images; got {mime} for {path.name}")
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def text_to_3d(
    prompt: str,
    art_style: Optional[str] = None,
    model_type: str = "lowpoly",
    target_formats: Optional[list] = None,
    ai_model: str = "meshy-6",
    out_dir: Path = Path("generated"),
    out_name: Optional[str] = None,
    refine: bool = True,
    enable_pbr: bool = False,
    texture_prompt: Optional[str] = None,
    target_polycount: Optional[int] = None,
    topology: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate a 3D model from a text prompt.

    Two-step per the API: a `preview` task builds untextured geometry, then a
    `refine` task (given the preview's id) textures it. Set refine=False to keep
    the untextured preview, which is faster and cheaper.

    model_type="lowpoly" yields game-ready clean polys; target_formats=["glb"]
    keeps completion time down by not building formats you won't use.
    """
    if not prompt or not prompt.strip():
        raise MeshyError("text_to_3d requires a non-empty prompt.")
    target_formats = target_formats or ["glb"]
    deadline = time.monotonic() + MAX_TOTAL_WAIT_SECONDS
    task_ids: list = []
    credits = 0

    payload = {
        "mode": "preview",
        "prompt": prompt,
        "ai_model": ai_model,
        "model_type": model_type,
        "target_formats": target_formats,
    }
    if target_polycount is not None:
        payload["target_polycount"] = target_polycount
    if topology is not None:
        payload["topology"] = topology
    if art_style:
        # art_style is deprecated for meshy-6; only send it when asked for.
        if ai_model in ("meshy-6", "latest"):
            _log(on_progress, f"note: art_style={art_style!r} is deprecated for {ai_model}")
        payload["art_style"] = art_style

    _log(on_progress, f"creating preview task ({ai_model}, {model_type}) ...")
    preview_id = _post_task(TEXT_TO_3D_PATH, payload)
    task_ids.append(preview_id)
    _log(on_progress, f"preview task {preview_id}")

    preview_task = _poll_until_done(TEXT_TO_3D_PATH, preview_id, deadline, "preview", on_progress)
    credits += preview_task.get("consumed_credits") or 0

    final_task = preview_task
    if refine:
        refine_payload = {
            "mode": "refine",
            "preview_task_id": preview_id,
            "ai_model": ai_model,
            "target_formats": target_formats,
        }
        if enable_pbr:
            refine_payload["enable_pbr"] = True
        if texture_prompt:
            refine_payload["texture_prompt"] = texture_prompt

        _log(on_progress, "creating refine (texturing) task ...")
        refine_id = _post_task(TEXT_TO_3D_PATH, refine_payload)
        task_ids.append(refine_id)
        _log(on_progress, f"refine task {refine_id}")

        final_task = _poll_until_done(TEXT_TO_3D_PATH, refine_id, deadline, "refine", on_progress)
        credits += final_task.get("consumed_credits") or 0

    out_dir = Path(out_dir)
    name = out_name or f"meshy_{task_ids[-1]}.glb"
    glb_path = _download(_glb_url(final_task), out_dir / name, on_progress)
    _log(on_progress, f"done — {credits} credit(s) used")

    return {
        "glb_path": glb_path,
        "task_ids": task_ids,
        "credits_used": credits,
        "thumbnail_url": final_task.get("thumbnail_url"),
        "model_urls": final_task.get("model_urls", {}),
        "task": final_task,
    }


def image_to_3d(
    image_url_or_path,
    ai_model: str = "meshy-6",
    model_type: str = "standard",
    target_formats: Optional[list] = None,
    should_texture: bool = True,
    enable_pbr: bool = False,
    texture_prompt: Optional[str] = None,
    target_polycount: Optional[int] = None,
    topology: Optional[str] = None,
    out_dir: Path = Path("generated"),
    out_name: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate a 3D model from an image (URL, data URI, or local .jpg/.png).

    Single-step: unlike text_to_3d there is no preview/refine loop.

    Note model_type="lowpoly" is DEPRECATED for image-to-3d (the valid values
    are "standard" and "smart-topology"), so it is not the default here; pass
    target_polycount instead to keep the mesh light.
    """
    target_formats = target_formats or ["glb"]
    deadline = time.monotonic() + MAX_TOTAL_WAIT_SECONDS

    if model_type == "lowpoly":
        _log(on_progress, "note: model_type='lowpoly' is deprecated for image-to-3d; "
                          "use target_polycount to control density")

    payload = {
        "image_url": _as_image_url(image_url_or_path),
        "ai_model": ai_model,
        "model_type": model_type,
        "target_formats": target_formats,
        "should_texture": should_texture,
    }
    if enable_pbr:
        payload["enable_pbr"] = True
    if texture_prompt:
        payload["texture_prompt"] = texture_prompt
    if target_polycount is not None:
        payload["target_polycount"] = target_polycount
    if topology is not None:
        payload["topology"] = topology

    _log(on_progress, f"creating image-to-3d task ({ai_model}, {model_type}) ...")
    task_id = _post_task(IMAGE_TO_3D_PATH, payload)
    _log(on_progress, f"task {task_id}")

    task = _poll_until_done(IMAGE_TO_3D_PATH, task_id, deadline, "image-to-3d", on_progress)
    credits = task.get("consumed_credits") or 0

    out_dir = Path(out_dir)
    name = out_name or f"meshy_{task_id}.glb"
    glb_path = _download(_glb_url(task), out_dir / name, on_progress)
    _log(on_progress, f"done — {credits} credit(s) used")

    return {
        "glb_path": glb_path,
        "task_ids": [task_id],
        "credits_used": credits,
        "thumbnail_url": task.get("thumbnail_url"),
        "model_urls": task.get("model_urls", {}),
        "task": task,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a 3D model via the Meshy API.")
    parser.add_argument("--prompt", type=str, help="Text prompt (text-to-3d).")
    parser.add_argument("--image", type=str, help="Image URL or local path (image-to-3d).")
    parser.add_argument("--out-dir", type=Path, default=Path("generated"))
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--art-style", type=str, default=None)
    parser.add_argument("--no-refine", action="store_true", help="Skip texturing (text-to-3d).")
    args = parser.parse_args()

    if not args.prompt and not args.image:
        parser.error("one of --prompt or --image is required")

    try:
        if args.prompt:
            res = text_to_3d(
                args.prompt,
                art_style=args.art_style,
                model_type=args.model_type or "lowpoly",
                out_dir=args.out_dir,
                refine=not args.no_refine,
            )
        else:
            res = image_to_3d(
                args.image,
                model_type=args.model_type or "standard",
                out_dir=args.out_dir,
            )
    except MeshyError as e:
        raise SystemExit(f"ERROR: {e}")

    print(f"GLB:     {res['glb_path']}")
    print(f"Tasks:   {res['task_ids']}")
    print(f"Credits: {res['credits_used']}")
