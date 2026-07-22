# 3D Asset Generator

Turns a text prompt and/or a reference image into a downloadable `.glb` model,
via a local Flask dashboard or the command line. Two backends, pick per
generation:

| Backend | Speed | Quality | Output | Needs |
|---|---|---|---|---|
| **Meshy 6** (primary) | ~2 min | Textured, production-quality game assets | `.glb` | `MESHY_API_KEY` |
| **Blender LLM pipeline** (secondary) | Slower | Blocky primitive assembly | `.glb` + editable `.py` | `ANTHROPIC_API_KEY` |

**Use Meshy for actual game assets.** The Blender backend does not produce
production meshes — it assembles cubes/cylinders/spheres/cones/tori via an
LLM-written Blender script, so output is prototype-grade and parts may not
fully connect. Its advantage is the `.py` it hands back: real, editable
Blender code you can open and parameterize by hand, which a Meshy `.glb`
doesn't give you.

## How the Blender backend works

Three models split the work so no model grades its own output:

1. **Claude Sonnet 5** (art director) turns the prompt/image into a structured
   build spec — parts, counts, palette, procedural approach.
2. **Claude Opus 4.8** (coder) writes a Blender Python script from the spec.
3. Blender renders a preview headlessly.
4. **Claude Sonnet 5** (critic) compares the render against the spec and
   returns structured diffs — a *different* model than the coder, enforced by
   an assertion, since a model critiquing its own code rubber-stamps it.
5. The coder revises from the diffs. Repeat for `iters` rounds, then export.

Only a script that actually rendered is ever exported (a crash on the last
round falls back to the last good render), and since there's no reference to
score against, every iteration's preview is kept so you can compare and pick
one instead of assuming the last is best.

## Install

    pip install flask requests anthropic google-genai lpips torch pillow scikit-image rembg "rembg[cpu]" scipy

Also install **Blender 4.2+** for the Blender backend. If it isn't on PATH,
set `BLENDER_BIN` to the blender binary.

## Environment variables

    export MESHY_API_KEY=...       # Meshy backend
    export ANTHROPIC_API_KEY=...   # Blender backend (Sonnet + Opus)
    # Windows: set MESHY_API_KEY=...  /  set ANTHROPIC_API_KEY=...

`GEMINI_API_KEY` is accepted by the model-calling code but not required —
everything currently runs on Claude models. See `.env.example`. Only the key(s)
for the backend(s) you actually use are required; the dashboard shows which
are set.

## Dashboard

    python app.py

Opens at **http://localhost:5000**. On one page:

- **Backend toggle** — Meshy or Blender. Switching reveals that backend's options.
- **Prompt box** + optional image upload (style reference / image-to-3D input).
  A prompt, an image, or both work; be concrete about part counts and
  proportions ("9 fronds, thick tapered trunk") — that's what the art
  director and critic actually check against.
- **Meshy options**: art style (leave default for meshy-6 — it's deprecated
  there), low-poly toggle, refine/texture pass (slower, more credits), PBR maps.
- **Blender options**: iteration count slider, a "let me pick which iteration
  to export" checkbox, and coder/critic model selectors (must differ).
- **Generate** — runs in a background thread; the page polls and streams the
  log live.
- **Result** — a `<model-viewer>` embed you can orbit, plus GLB download (and
  `.py` + iteration contact-sheet download for Blender runs).
- **Iteration picker** (Blender, when enabled) — every iteration's preview
  side by side with the critic's diff count, so you can eyeball which one
  actually looks best rather than trusting the last round blindly.
- **Asset library** — every generated model, with its prompt, backend, and
  timestamp, viewable and downloadable. Backed by a JSON index on disk
  (`generated/library.json`), no database.

## CLI alternatives

Both backends also run standalone, without the dashboard.

**Meshy:**

    python meshy.py --prompt "low-poly palm tree, stylized, game-ready" --out-dir generated
    python meshy.py --image chair.jpg --out-dir generated

**Blender pipeline:**

    python generate.py --prompt "low-poly palm tree, stylized, game-ready" --out palm.glb --iters 3
    python generate.py --style-image chair.jpg --out chair.glb --pick
    python generate.py --style-image chair.jpg --prompt "make it autumnal" --out chair.glb

`--pick` pauses after the loop and lets you choose which rendered iteration to
export, from the contact sheet, rather than defaulting to the last one.
`--coder` / `--critic` / `--director` override the default models (Opus /
Sonnet / Sonnet); `--coder` and `--critic` must differ.

## Also in this repo: benchmark.py

`benchmark.py` is a **separate experiment**, not part of the asset-generation
workflow above — it runs the Blender pipeline across several (coder, critic)
model pairs against the same reference image and scores each on silhouette
IoU, to compare model quality. Not needed to generate assets:

    python benchmark.py --ref chair.jpg --max-iters 5
