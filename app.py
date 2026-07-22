"""app.py — a localhost dashboard for both 3D generation backends.

    Meshy   — fast, textured, production quality (the default)
    Blender — slower and blockier, but emits editable Python you can parameterize

Run:
    python app.py            # http://localhost:5000

Generation runs in a background thread; the page polls /api/job/<id> for status
and streamed log lines. The Blender backend can pause mid-run to let you pick
which iteration to export (the web equivalent of generate.py's --pick).

Reads MESHY_API_KEY and ANTHROPIC_API_KEY from the environment. No database —
the asset library is a JSON index on disk.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, render_template, abort

import meshy
import generate as blender_backend


APP_ROOT = Path(__file__).resolve().parent
GENERATED_DIR = APP_ROOT / "generated"
LIBRARY_INDEX = GENERATED_DIR / "library.json"
UPLOAD_DIR = GENERATED_DIR / "uploads"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB uploads

# job_id -> job dict. In-process only; jobs do not survive a restart.
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_LIBRARY_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Job bookkeeping
# --------------------------------------------------------------------------

def _new_job(backend: str, prompt: str) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "backend": backend,
        "prompt": prompt,
        "status": "running",          # running | awaiting_pick | done | error
        "log": [],
        "error": None,
        "result": None,               # {glb_url, script_url, ...}
        "candidates": None,           # Blender: [{iter, render_url, n_diffs}]
        "contact_sheet_url": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Set when the Blender run pauses for an iteration choice.
        "_pick_event": threading.Event(),
        "_picked_iter": None,
    }
    with _JOBS_LOCK:
        _JOBS[job["id"]] = job
    return job


def _log(job: dict, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    job["log"].append(f"[{stamp}] {message}")


def _public(job: dict) -> dict:
    """Job view safe to serialize (drops threading primitives)."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# Asset library (JSON index, no database)
# --------------------------------------------------------------------------

def _load_library() -> list:
    if not LIBRARY_INDEX.exists():
        return []
    try:
        return json.loads(LIBRARY_INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _add_to_library(entry: dict) -> None:
    with _LIBRARY_LOCK:
        lib = _load_library()
        lib.insert(0, entry)  # newest first
        LIBRARY_INDEX.write_text(json.dumps(lib, indent=2), encoding="utf-8")


def _asset_url(path) -> str:
    """Map a file under generated/ to its /files/<relpath> URL."""
    rel = Path(path).resolve().relative_to(GENERATED_DIR.resolve())
    return "/files/" + rel.as_posix()


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

def _run_meshy(job: dict, opts: dict) -> None:
    on_progress = lambda m: _log(job, m)
    stem = f"meshy_{job['id']}"

    if opts.get("image_path"):
        _log(job, "Meshy image-to-3d starting ...")
        res = meshy.image_to_3d(
            opts["image_path"],
            model_type=opts.get("model_type") or "standard",
            enable_pbr=opts.get("enable_pbr", False),
            texture_prompt=opts.get("prompt") or None,
            out_dir=GENERATED_DIR,
            out_name=f"{stem}.glb",
            on_progress=on_progress,
        )
    else:
        _log(job, "Meshy text-to-3d starting ...")
        res = meshy.text_to_3d(
            opts["prompt"],
            art_style=opts.get("art_style") or None,
            model_type="lowpoly" if opts.get("lowpoly", True) else "standard",
            enable_pbr=opts.get("enable_pbr", False),
            refine=opts.get("refine", True),
            out_dir=GENERATED_DIR,
            out_name=f"{stem}.glb",
            on_progress=on_progress,
        )

    job["result"] = {
        "glb_url": _asset_url(res["glb_path"]),
        "glb_name": Path(res["glb_path"]).name,
        "credits_used": res["credits_used"],
        "task_ids": res["task_ids"],
    }
    _add_to_library({
        "id": job["id"],
        "prompt": job["prompt"],
        "backend": "meshy",
        "created_at": job["created_at"],
        "glb_url": job["result"]["glb_url"],
        "glb_name": job["result"]["glb_name"],
        "script_url": None,
        "credits_used": res["credits_used"],
    })


def _run_blender(job: dict, opts: dict) -> None:
    out_glb = GENERATED_DIR / f"blender_{job['id']}.glb"

    def choose(candidates, contact_sheet):
        """Pause the run and let the page pick an iteration (the --pick equivalent)."""
        job["candidates"] = [
            {"iter": c["iter"], "render_url": _asset_url(c["render"]), "n_diffs": c["n_diffs"]}
            for c in candidates
        ]
        job["contact_sheet_url"] = _asset_url(contact_sheet) if contact_sheet else None
        job["status"] = "awaiting_pick"
        _log(job, f"Waiting for you to pick which of {len(candidates)} iteration(s) to export ...")

        # Bounded wait so a closed tab can't wedge the thread forever.
        if not job["_pick_event"].wait(timeout=600):
            _log(job, "No choice made within 10 min — exporting the default (latest).")
            job["status"] = "running"
            return candidates[-1]["iter"]

        picked = job["_picked_iter"]
        if picked is None or not any(c["iter"] == picked for c in candidates):
            picked = candidates[-1]["iter"]
        _log(job, f"Exporting iteration {picked}.")
        job["status"] = "running"
        return picked

    # generate.py prints progress; mirror it into the job log too.
    _log(job, f"Blender pipeline starting ({opts.get('iters', 3)} iters, "
              f"coder={opts.get('coder')}, critic={opts.get('critic')}) ...")

    res = blender_backend.generate(
        out_glb,
        prompt=opts.get("prompt") or None,
        style_image=opts.get("image_path") or None,
        iters=int(opts.get("iters", 3)),
        coder=opts.get("coder") or blender_backend.DEFAULT_CODER,
        critic=opts.get("critic") or blender_backend.DEFAULT_CRITIC,
        director=opts.get("director") or blender_backend.DEFAULT_DIRECTOR,
        choose=choose if opts.get("pick") else None,
    )

    script_path = out_glb.with_suffix(".py")
    sheet = res.get("contact_sheet")
    job["contact_sheet_url"] = _asset_url(sheet) if sheet else None
    job["candidates"] = [
        {"iter": c["iter"], "render_url": _asset_url(c["render"]), "n_diffs": c["n_diffs"]}
        for c in res.get("candidates", [])
    ]
    job["result"] = {
        "glb_url": _asset_url(res["glb"]),
        "glb_name": Path(res["glb"]).name,
        "script_url": _asset_url(script_path) if script_path.exists() else None,
        "script_name": script_path.name,
        "best_iter": res["best_iter"],
        "spec": res["spec"],
    }
    _add_to_library({
        "id": job["id"],
        "prompt": job["prompt"],
        "backend": "blender",
        "created_at": job["created_at"],
        "glb_url": job["result"]["glb_url"],
        "glb_name": job["result"]["glb_name"],
        "script_url": job["result"]["script_url"],
        "credits_used": None,
    })


def _run_job(job: dict, opts: dict) -> None:
    try:
        if job["backend"] == "meshy":
            _run_meshy(job, opts)
        else:
            _run_blender(job, opts)
        job["status"] = "done"
        _log(job, "Finished.")
    except Exception as e:  # surface any backend failure to the page
        job["status"] = "error"
        job["error"] = str(e)
        _log(job, f"ERROR: {e}")
        traceback.print_exc()
    finally:
        # Never leave a paused job blocking a waiter.
        job["_pick_event"].set()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        has_meshy=bool(os.environ.get("MESHY_API_KEY") or os.environ.get("MESHI_API_KEY")),
        has_anthropic=bool(os.environ.get("ANTHROPIC_API_KEY")),
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    backend = (request.form.get("backend") or "meshy").lower()
    prompt = (request.form.get("prompt") or "").strip()

    image_path = None
    upload = request.files.get("image")
    if upload and upload.filename:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            return jsonify({"error": f"Unsupported image type {suffix}; use .jpg or .png"}), 400
        image_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
        upload.save(image_path)

    if not prompt and not image_path:
        return jsonify({"error": "Provide a prompt, an image, or both."}), 400

    if backend == "meshy":
        if not (os.environ.get("MESHY_API_KEY") or os.environ.get("MESHI_API_KEY")):
            return jsonify({"error": "MESHY_API_KEY is not set in the environment."}), 400
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify({"error": "ANTHROPIC_API_KEY is not set in the environment."}), 400
        coder = request.form.get("coder") or blender_backend.DEFAULT_CODER
        critic = request.form.get("critic") or blender_backend.DEFAULT_CRITIC
        if coder == critic:
            return jsonify({"error": "Coder and critic must be different models."}), 400

    opts = {
        "prompt": prompt,
        "image_path": str(image_path) if image_path else None,
        "art_style": request.form.get("art_style"),
        "lowpoly": request.form.get("lowpoly") == "true",
        "enable_pbr": request.form.get("enable_pbr") == "true",
        "refine": request.form.get("refine") != "false",
        "iters": request.form.get("iters", 3),
        "pick": request.form.get("pick") == "true",
        "coder": request.form.get("coder"),
        "critic": request.form.get("critic"),
        "director": request.form.get("director"),
    }

    job = _new_job(backend, prompt or f"(image: {Path(image_path).name})")
    threading.Thread(target=_run_job, args=(job, opts), daemon=True).start()
    return jsonify({"job_id": job["id"]})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = _JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(_public(job))


@app.route("/api/job/<job_id>/pick", methods=["POST"])
def api_pick(job_id):
    job = _JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "awaiting_pick":
        return jsonify({"error": "job is not awaiting a pick"}), 409

    data = request.get_json(silent=True) or {}
    try:
        picked = int(data.get("iter"))
    except (TypeError, ValueError):
        return jsonify({"error": "iter must be an integer"}), 400

    job["_picked_iter"] = picked
    job["_pick_event"].set()
    return jsonify({"ok": True, "picked": picked})


@app.route("/api/library")
def api_library():
    return jsonify(_load_library())


@app.route("/files/<path:relpath>")
def files(relpath):
    """Serve anything under generated/ (GLBs, previews, contact sheets, scripts)."""
    full = (GENERATED_DIR / relpath).resolve()
    # Contain path traversal: the resolved path must stay inside generated/.
    if not str(full).startswith(str(GENERATED_DIR.resolve())) or not full.is_file():
        abort(404)
    as_download = request.args.get("download") == "1"
    return send_from_directory(GENERATED_DIR, relpath, as_attachment=as_download)


if __name__ == "__main__":
    print("Backends: Meshy " + ("[key set]" if os.environ.get("MESHY_API_KEY") or os.environ.get("MESHI_API_KEY") else "[NO KEY]")
          + " | Blender " + ("[key set]" if os.environ.get("ANTHROPIC_API_KEY") else "[NO KEY]"))
    print("Dashboard: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
