"""Model-calling wrappers for the render benchmark loop.

Three entry points:
  gen_spec(ref_image_path, spec_model)        -> (dict, usage)
  gen_script(spec, prev_script, diffs,
             render_grid_path, claude_model)  -> (str, usage)
  gen_critique(ref_grid_path, render_grid_path,
               spec, critique_model)          -> (list[dict], usage)

Model names are always parameters, never hardcoded — this code benchmarks
several models against each other, so a hardcoded name would silently
invalidate the experiment.

gen_spec and gen_critique dispatch on the model string: names starting with
"claude-" route to the Anthropic SDK, everything else routes to google-genai.
Gemini API access is currently unavailable (project-level 403, no billing in
this region), so today both roles run on Claude in practice — but the
google-genai code path is kept intact behind the dispatcher so it can be
switched back on later without a rewrite. Both paths produce identical
output shapes.

gen_script is Claude-only by design (it's the one role being benchmarked
against a fixed critic/spec pipeline), so it takes no dispatcher.

Reads API keys from env: ANTHROPIC_API_KEY, GEMINI_API_KEY.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

import anthropic
from google import genai
from google.genai import types as genai_types


_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _media_type_for(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image extension '{suffix}' for {path}")
    return media_type


def _read_image_bytes(path: Path) -> tuple[bytes, str]:
    """Return (raw_bytes, media_type) for an image file."""
    media_type = _media_type_for(path)
    return Path(path).read_bytes(), media_type


def _read_image_b64(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file — used for the Anthropic SDK."""
    data, media_type = _read_image_bytes(path)
    return base64.standard_b64encode(data).decode("ascii"), media_type


def _anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=api_key)


def _genai_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def _strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing ``` or ```lang ... ``` fences if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (``` or ```lang).
    lines = lines[1:]
    # Drop a trailing fence line if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_strict(text: str):
    return json.loads(_strip_markdown_fences(text))


# --------------------------------------------------------------------------
# Gemini calls (spec generation, critique) with JSON-parse retry
# --------------------------------------------------------------------------

def _call_gemini_json(model: str, parts: list, retry_reminder: str):
    """Call Gemini requesting JSON-only output; retry once on parse failure.

    Returns (parsed_json, usage_dict). Raises on second failure.
    """
    client = _genai_client()
    config = genai_types.GenerateContentConfig(response_mime_type="application/json")

    response = client.models.generate_content(model=model, contents=parts, config=config)
    text = response.text or ""
    usage = _gemini_usage(response)

    try:
        return _parse_json_strict(text), usage
    except (json.JSONDecodeError, ValueError):
        pass

    retry_parts = list(parts) + [retry_reminder]
    response = client.models.generate_content(model=model, contents=retry_parts, config=config)
    text = response.text or ""
    usage = _accumulate_usage(usage, _gemini_usage(response))

    return _parse_json_strict(text), usage


def _gemini_usage(response) -> dict:
    meta = getattr(response, "usage_metadata", None)
    in_tokens = getattr(meta, "prompt_token_count", None) or 0
    out_tokens = getattr(meta, "candidates_token_count", None) or 0
    return {"in": in_tokens, "out": out_tokens}


def _accumulate_usage(a: dict, b: dict) -> dict:
    return {"in": a["in"] + b["in"], "out": a["out"] + b["out"]}


def _anthropic_usage(response) -> dict:
    return {"in": response.usage.input_tokens, "out": response.usage.output_tokens}


def _is_claude_model(model: str) -> bool:
    return model.startswith("claude-")


# --------------------------------------------------------------------------
# Anthropic structured output via forced tool use
#
# The Anthropic API has no response_mime_type, and Sonnet 5 rejects assistant
# message prefill. The robust way to get structured JSON is forced tool use:
# define a tool whose input_schema is the target shape, then force the model
# to call it with tool_choice={"type": "tool", "name": ...}. The result comes
# back already parsed in the tool_use block's `input` field — no string
# parsing, no fences, no malformed-JSON retry to handle.
# --------------------------------------------------------------------------

def _call_claude_tool(model: str, system_prompt: str, user_content: list, tool: dict) -> tuple[dict, dict]:
    """Call Claude with a single forced tool; return (tool_input, usage_dict).

    tool_input is the already-parsed `input` of the tool_use block.
    """
    client = _anthropic_client()

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_input = None
    for block in response.content:
        if block.type == "tool_use" and block.name == tool["name"]:
            tool_input = block.input
            break
    if tool_input is None:
        raise ValueError(f"Claude did not return a '{tool['name']}' tool_use block as forced")

    return tool_input, _anthropic_usage(response)


# --------------------------------------------------------------------------
# 1) gen_spec
# --------------------------------------------------------------------------

_SPEC_PROMPT = """Analyze the reference image and describe the object as a structured \
part decomposition for procedural 3D reconstruction in Blender.

Return JSON ONLY. No prose, no markdown fences, no commentary — just the JSON object.

Schema:
{
  "object": str,
  "parts": [{"name": str,
             "primitive": "cube"|"cylinder"|"sphere"|"cone"|"torus",
             "approx_dims": [x, y, z],
             "position": [x, y, z],
             "rotation_euler": [x, y, z],
             "notes": str}],
  "relations": [str],
  "overall_scale": float
}

Rules:
- "primitive" must be one of exactly: cube, cylinder, sphere, cone, torus.
- approx_dims are in local/object space (meters), not world space.
- position and rotation_euler are in world space; rotation_euler is in radians.
- "relations" is a list of short free-text notes about how parts connect \
(e.g. "leg_1 attaches to seat_bottom at its top face").
- "overall_scale" is a single multiplier suggesting the overall size of the \
object in meters (its largest dimension).
- Decompose into as many primitive parts as needed to reasonably approximate \
the object's shape.
- Parts that connect (legs to seat, rails to crossbars) must OVERLAP at their \
joints, not just touch — position and size each part so it extends slightly \
into its neighbor, so the exported mesh reads as one connected object rather \
than floating pieces. Prefer slight interpenetration over gaps.
"""

_JSON_RETRY_REMINDER = "Return valid JSON only. No markdown fences, no prose, no explanation — just the JSON."

_PRIMITIVE_ENUM = ["cube", "cylinder", "sphere", "cone", "torus"]

# Forced-tool schema for gen_spec on the Claude path. input_schema mirrors the
# _SPEC_PROMPT JSON shape so the model's tool call arrives already parsed.
_EMIT_SPEC_TOOL = {
    "name": "emit_spec",
    "description": "Emit the structured part decomposition of the object in the reference image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "object": {"type": "string"},
            "parts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "primitive": {"type": "string", "enum": _PRIMITIVE_ENUM},
                        "approx_dims": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "position": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "rotation_euler": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "notes": {"type": "string"},
                    },
                    "required": ["name", "primitive", "approx_dims", "position", "rotation_euler", "notes"],
                },
            },
            "relations": {"type": "array", "items": {"type": "string"}},
            "overall_scale": {"type": "number"},
        },
        "required": ["object", "parts", "relations", "overall_scale"],
    },
}


def gen_spec(ref_image_path: Path, spec_model: str) -> tuple[dict, dict]:
    """Analyze a reference image and produce a structured part spec.

    Dispatches on spec_model: "claude-*" routes to Anthropic, anything else
    routes to google-genai (Gemini). Both paths return the same shape.
    """
    if _is_claude_model(spec_model):
        return _gen_spec_claude(ref_image_path, spec_model)
    return _gen_spec_gemini(ref_image_path, spec_model)


def _gen_spec_gemini(ref_image_path: Path, gemini_model: str) -> tuple[dict, dict]:
    ref_image_path = Path(ref_image_path)
    raw_bytes, media_type = _read_image_bytes(ref_image_path)

    image_part = genai_types.Part.from_bytes(data=raw_bytes, mime_type=media_type)
    parts = [image_part, _SPEC_PROMPT]

    spec, usage = _call_gemini_json(gemini_model, parts, _JSON_RETRY_REMINDER)
    return spec, usage


def _gen_spec_claude(ref_image_path: Path, claude_model: str) -> tuple[dict, dict]:
    ref_image_path = Path(ref_image_path)
    b64_data, media_type = _read_image_b64(ref_image_path)

    user_content = [
        {"type": "text", "text": _SPEC_PROMPT},
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
    ]

    system_prompt = "You analyze reference images and report structured 3D part decompositions via the emit_spec tool."
    spec, usage = _call_claude_tool(claude_model, system_prompt, user_content, _EMIT_SPEC_TOOL)
    return spec, usage


# --------------------------------------------------------------------------
# 1b) gen_spec_from_prompt — the "art director" for text-to-asset mode
# --------------------------------------------------------------------------

_ART_DIRECTOR_TEMPLATE = """You are an art director for STYLIZED LOW-POLY GAME ASSETS. Turn the \
brief below into a detailed, buildable spec for a procedural Blender script.

{brief}

This is a stylized game asset, NOT a photo reconstruction. Prioritize a clean, readable \
LOW-POLY SILHOUETTE and appealing proportions over real-world accuracy. Exaggerate for \
readability where it helps the shape read at a glance.

Your spec must be concrete enough to code directly from:
- Give every repeated part an explicit integer COUNT (e.g. 9 fronds, 6 trunk segments) — \
never "some" or "several".
- Give a PALETTE of hex colors with names, and assign each part one palette color by name.
- Give PROPORTIONS as concrete ratios or sizes (e.g. "trunk height 4x its base radius", \
"frond length 1.5x trunk height").
- Describe a PROCEDURAL approach: repeated parts MUST be generated with loops/arrays and \
math (radial placement via sin/cos around an axis, stacked tapered segments via a loop \
that shrinks radius per step) — never hand-placed one-off primitives.
- Keep the polycount low: prefer few-sided primitives (6-8 sided cylinders, low-segment \
spheres) and state a polycount target.
- Parts that connect must OVERLAP at their joints (slight interpenetration, never gaps) so \
the exported mesh reads as one connected object.

Return JSON ONLY. No prose, no markdown fences, no commentary — just the JSON object.

Schema:
{{
  "object": str,
  "style_notes": str,
  "palette": [{{"name": str, "hex": str}}],
  "parts": [{{"name": str,
             "count": int,
             "primitive": "cube"|"cylinder"|"sphere"|"cone"|"torus",
             "proportions": str,
             "placement": str,
             "color": str,
             "procedural_notes": str}}],
  "procedural_approach": str,
  "polycount_target": str,
  "overall_scale": float
}}

Rules:
- "primitive" must be one of exactly: cube, cylinder, sphere, cone, torus.
- "count" is a specific integer (use 1 for non-repeated parts).
- "hex" is a "#RRGGBB" string.
- "color" must match one of the palette entry names.
- "placement" says where the part goes and how repeats are distributed (e.g. "radially \
around the trunk top, evenly spaced, each tilted 40 degrees downward and drooping").
- "procedural_notes" gives the coder the actual loop/math strategy for this part (e.g. \
"for i in range(count): angle = 2*pi*i/count; x = r*cos(angle); y = r*sin(angle)").
- "overall_scale" is the asset's largest dimension in meters.
"""

# Forced-tool schema for gen_spec_from_prompt on the Claude path.
_EMIT_BUILD_SPEC_TOOL = {
    "name": "emit_build_spec",
    "description": "Emit the art-directed build spec for a stylized low-poly game asset.",
    "input_schema": {
        "type": "object",
        "properties": {
            "object": {"type": "string"},
            "style_notes": {"type": "string"},
            "palette": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "hex": {"type": "string"},
                    },
                    "required": ["name", "hex"],
                },
            },
            "parts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                        "primitive": {"type": "string", "enum": _PRIMITIVE_ENUM},
                        "proportions": {"type": "string"},
                        "placement": {"type": "string"},
                        "color": {"type": "string"},
                        "procedural_notes": {"type": "string"},
                    },
                    "required": [
                        "name", "count", "primitive", "proportions",
                        "placement", "color", "procedural_notes",
                    ],
                },
            },
            "procedural_approach": {"type": "string"},
            "polycount_target": {"type": "string"},
            "overall_scale": {"type": "number"},
        },
        "required": [
            "object", "style_notes", "palette", "parts",
            "procedural_approach", "polycount_target", "overall_scale",
        ],
    },
}


_STYLE_IMAGE_CAVEAT = (
    "CRITICAL: the attached image is STYLE INSPIRATION, NOT a reconstruction target. Do NOT "
    "attempt a pixel-accurate or photo-real copy of it. Reinterpret its subject as clean, "
    "stylized low-poly game art with a readable silhouette — simplify detail away, keep the "
    "shape language and character."
)


def _build_art_director_brief(prompt: Optional[str], has_style_image: bool) -> str:
    """Assemble the brief for the three input modes: text, image, or both."""
    if has_style_image and prompt:
        brief = (
            "BRIEF: The attached image is STYLE INSPIRATION — take your cue from its subject, "
            "shape language, proportions, and palette.\n\n"
            f"ADDITIONAL DIRECTION: {prompt}\n\n"
            "The image guides the style; the text adds direction. Where the two conflict, "
            "follow the text."
        )
    elif has_style_image:
        brief = (
            "BRIEF: The attached image is STYLE INSPIRATION. Build a stylized low-poly game "
            "asset of its subject, taking your cue from its shape language, proportions, and "
            "palette."
        )
    elif prompt:
        brief = f"BRIEF / DESCRIPTION: {prompt}"
    else:
        raise ValueError("gen_spec_from_prompt requires a prompt, a style_image, or both")

    if has_style_image:
        brief += "\n\n" + _STYLE_IMAGE_CAVEAT
    return brief


def gen_spec_from_prompt(
    prompt: Optional[str],
    spec_model: str,
    style_image: Optional[Path] = None,
) -> tuple[dict, dict]:
    """Art-direct a text description and/or a style image into a low-poly asset spec.

    Three input modes (at least one of prompt/style_image is required):
      prompt only       -> spec invented from the text
      style_image only  -> spec derived from the image AS STYLE INSPIRATION
      both              -> image guides style, text adds direction

    The image is never treated as a reconstruction target — this produces a
    stylized low-poly asset, not a photo copy.

    Dispatches on spec_model: "claude-*" routes to Anthropic, anything else
    routes to google-genai (Gemini). Both paths return the same shape.
    """
    if prompt is None and style_image is None:
        raise ValueError("gen_spec_from_prompt requires a prompt, a style_image, or both")

    text = _ART_DIRECTOR_TEMPLATE.format(
        brief=_build_art_director_brief(prompt, style_image is not None)
    )
    if _is_claude_model(spec_model):
        return _gen_spec_from_prompt_claude(text, spec_model, style_image)
    return _gen_spec_from_prompt_gemini(text, spec_model, style_image)


def _gen_spec_from_prompt_gemini(
    text: str, gemini_model: str, style_image: Optional[Path]
) -> tuple[dict, dict]:
    parts: list = [text]
    if style_image is not None:
        data, media_type = _read_image_bytes(Path(style_image))
        parts += ["STYLE REFERENCE:", genai_types.Part.from_bytes(data=data, mime_type=media_type)]
    return _call_gemini_json(gemini_model, parts, _JSON_RETRY_REMINDER)


def _gen_spec_from_prompt_claude(
    text: str, claude_model: str, style_image: Optional[Path]
) -> tuple[dict, dict]:
    user_content: list = [{"type": "text", "text": text}]
    if style_image is not None:
        b64, media_type = _read_image_b64(Path(style_image))
        user_content.append({"type": "text", "text": "STYLE REFERENCE:"})
        user_content.append(
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
        )
    system_prompt = (
        "You are an art director for stylized low-poly game assets. You turn text descriptions "
        "and/or style-reference images into concrete, procedurally buildable specs via the "
        "emit_build_spec tool. Style images are inspiration, never reconstruction targets."
    )
    return _call_claude_tool(claude_model, system_prompt, user_content, _EMIT_BUILD_SPEC_TOOL)


# --------------------------------------------------------------------------
# 2) gen_script
# --------------------------------------------------------------------------

_SCRIPT_SYSTEM_PROMPT = """You are generating a Blender Python script that creates GEOMETRY ONLY.

The script will be exec()'d inside a harness that owns the camera, lights, \
render engine, resolution, color management, and rendering itself. The harness \
clears the scene to an empty factory state before running your script, and will \
discard anything your script sets outside of geometry.

Your script MUST NOT:
- create or move a camera
- create or configure lights
- set the render engine
- set resolution, samples, or any render/image settings
- call bpy.ops.render.render or any render-triggering operator
- call bpy.ops.wm.save_mainfile or any save operator
- clear or reset the scene (the harness already does this)

Your script SHOULD:
- use bpy.ops.mesh primitive operators (e.g. bpy.ops.mesh.primitive_cube_add, \
primitive_cylinder_add, primitive_uv_sphere_add, primitive_cone_add, \
primitive_torus_add) to create each part
- set object transforms (location, rotation_euler, scale) directly on the \
resulting bpy.context.object, or via the operator's location/rotation kwargs
- target the Blender 4.x/5.x common API surface — avoid version-specific \
enums or properties that differ between Blender releases
- name objects sensibly so parts are identifiable
- make parts that connect (legs to seat, rails to crossbars) OVERLAP at their \
joints, not just touch — extend each part slightly into its neighbor so the \
exported mesh reads as one connected object rather than floating pieces; prefer \
slight interpenetration over gaps

Output format: return the COMPLETE Python script as plain text. Do not wrap it \
in markdown code fences. Do not include any prose before or after the code — \
output only the script itself, from the first import to the last line.
"""


_PROCEDURAL_SCRIPT_SYSTEM_PROMPT = """You are generating a Blender Python script that \
procedurally builds a STYLIZED LOW-POLY GAME ASSET: geometry AND simple materials.

The script will be exec()'d inside a harness that owns the camera, lights, render engine, \
resolution, and rendering/export itself. The harness clears the scene to an empty factory \
state before running your script.

Your script MUST NOT:
- create or move a camera
- create or configure lights
- set the render engine
- set resolution, samples, or any render/image settings
- call bpy.ops.render.render or any render-triggering operator
- call bpy.ops.wm.save_mainfile, bpy.ops.export_scene.*, or any save/export operator
- clear or reset the scene (the harness already does this)

Your script MUST:
- BUILD REPEATED PARTS PROCEDURALLY, with loops and math — never hand-place copies. Use the \
spec's integer counts. Radial placement comes from sin/cos (for i in range(n): angle = \
2*math.pi*i/n; x = r*math.cos(angle); y = r*math.sin(angle)); stacked/tapered parts come \
from a loop that steps position and shrinks radius/scale per segment. A hardcoded list of \
near-identical primitives is wrong — use the loop.
- ASSIGN THE SPEC'S PALETTE as simple materials. Create one bpy.data.materials per palette \
entry with use_nodes=True, set the Principled BSDF "Base Color", and append it to each \
object's data.materials. Reuse one material per palette color; do not create a material \
per object. Materials are exported to glTF, so they matter.
- CONVERT HEX sRGB TO LINEAR before assigning Base Color, or the exported colors will be \
wrong. Include a helper:
    def srgb_to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
  and build the color as tuple(srgb_to_linear(int(h[i:i+2], 16) / 255) for i in (1, 3, 5)) + (1.0,).
- KEEP THE POLYCOUNT LOW. Pass low segment counts to primitives (e.g. \
primitive_cylinder_add(vertices=6 or 8), primitive_uv_sphere_add(segments=8, ring_count=6), \
primitive_cone_add(vertices=6)). Respect the spec's polycount target. A clean low-poly \
silhouette beats detail.
- MAKE CONNECTED PARTS OVERLAP at their joints — extend each part slightly into its \
neighbor so the exported mesh reads as one connected object rather than floating pieces. \
Prefer slight interpenetration over gaps.

Your script SHOULD:
- use bpy.ops.mesh primitive operators (primitive_cube_add, primitive_cylinder_add, \
primitive_uv_sphere_add, primitive_cone_add, primitive_torus_add) to create each part
- set object transforms (location, rotation_euler, scale) on the resulting \
bpy.context.object, or via the operator's location/rotation kwargs
- target the Blender 4.x/5.x common API surface — avoid version-specific enums or \
properties that differ between Blender releases
- import math and name objects sensibly so parts are identifiable

Prioritize a clean, readable low-poly silhouette and appealing proportions over \
photo-accuracy. This is a stylized game asset.

Output format: return the COMPLETE Python script as plain text. Do not wrap it in markdown \
code fences. Do not include any prose before or after the code — output only the script \
itself, from the first import to the last line.
"""


def _build_script_user_prompt(spec: dict, prev_script: Optional[str], diffs: Optional[list]) -> str:
    sections = ["Target object spec (JSON):", json.dumps(spec, indent=2)]

    if prev_script is not None:
        sections.append("\nPrevious script (this is the current best attempt — patch it, don't start over):")
        sections.append(prev_script)

    if diffs is not None:
        sections.append("\nCritique of the previous render, as structured diffs (JSON):")
        sections.append(json.dumps(diffs, indent=2))
        sections.append(
            "\nAddress these diffs by adjusting the script's geometry, positions, "
            "rotations, and scales. Return the complete corrected script."
        )
    else:
        sections.append(
            "\nThis is the first attempt. Write a complete script from scratch that "
            "builds the object described by the spec."
        )

    return "\n".join(sections)


def gen_script(
    spec: dict,
    prev_script: Optional[str],
    diffs: Optional[list],
    render_grid_path: Optional[Path],
    claude_model: str,
    mode: str = "geometry",
) -> tuple[str, dict]:
    """Ask Claude to write or patch a Blender script.

    Stateless: each call is a single fresh request built from spec + prev_script +
    diffs. No chat history is accumulated across iterations.

    render_grid_path, if provided, is attached as an image so Claude can see the
    current render alongside the structured critique.

    mode selects the system prompt:
      "geometry" (default) — geometry-only, for the reference-image/benchmark path.
      "asset"              — procedural low-poly game asset with palette materials.
    """
    if mode not in ("geometry", "asset"):
        raise ValueError(f"gen_script mode must be 'geometry' or 'asset', got {mode!r}")
    system_prompt = _PROCEDURAL_SCRIPT_SYSTEM_PROMPT if mode == "asset" else _SCRIPT_SYSTEM_PROMPT

    client = _anthropic_client()

    user_content: list = [{"type": "text", "text": _build_script_user_prompt(spec, prev_script, diffs)}]

    if render_grid_path is not None:
        b64_data, media_type = _read_image_b64(Path(render_grid_path))
        user_content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
            }
        )

    response = client.messages.create(
        model=claude_model,
        max_tokens=16384,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    script = _strip_markdown_fences(text)
    usage = _anthropic_usage(response)

    return script, usage


# --------------------------------------------------------------------------
# 3) gen_critique
# --------------------------------------------------------------------------

_CRITIQUE_PROMPT_TEMPLATE = """You are grading a procedurally generated 3D model against a \
reference. Four images are provided below, each clearly labeled. Every image is a grid of \
the same 6 fixed views of the object: front, back, left, right, top, three-quarter — tiled \
left-to-right, top-to-bottom in that order.

  - REFERENCE PHOTO: a real photograph of the target object.
  - REFERENCE SILHOUETTE: the object's shape as a black-on-white mask.
  - RENDER: the current 3D model — UNTEXTURED, FLAT GRAY GEOMETRY under neutral lighting.
  - RENDER SILHOUETTE: the render's shape as a black-on-white mask.

CRITICAL — what you may and may not judge:
- The render is untextured gray geometry; the reference is a real photograph. This
  difference in color, material, texture, and lighting is EXPECTED and is NOT a defect.
- Judge ONLY: SHAPE, PROPORTION, and PLACEMENT of parts. Use the two SILHOUETTES as your
  primary evidence for shape, and the RGB images to disambiguate part structure.
- NEVER comment on color, material, texture, surface finish, shading, reflectivity, or
  lighting. The coder generates geometry only and cannot change any of those. Such critiques
  are invalid and waste iterations — do not emit them under any circumstances.

The object is meant to match this spec:
{spec_json}

Identify SHAPE/PROPORTION/PLACEMENT discrepancies. Return JSON ONLY — no prose, no markdown \
fences — as a list of objects with this schema:

[
  {{"part": str,
    "issue": str,
    "axis": "x"|"y"|"z"|"scale"|"missing"|"extra",
    "suggested_delta": str,
    "severity": "high"|"medium"|"low"}}
]

Rules:
- "part" should reference a part name from the spec when possible, or a short \
descriptive label if the discrepancy doesn't map to a named part.
- "axis" categorizes the kind of error: a positional/rotational error along a \
specific axis, a scale error, a part missing from the render, or an extra part \
in the render not in the reference.
- "suggested_delta" is a short, concrete, actionable instruction (e.g. "move \
+0.3 along y" or "increase radius by ~20%").
- "issue" must describe a shape/proportion/placement problem only — never color, \
material, texture, or lighting.
- Return an empty list [] if the shapes match well — do not invent issues to fill the list.
- Do not use free-text paragraphs; every entry must fit the schema exactly.
"""

# Forced-tool schema for gen_critique on the Claude path. Anthropic tool
# input_schema must be an object at the top level, so the list of diffs is
# wrapped under a "diffs" key here; _gen_critique_claude unwraps it so the
# public gen_critique still returns a bare list.
_EMIT_CRITIQUE_TOOL = {
    "name": "emit_critique",
    "description": "Emit the structured list of discrepancies between the render grid and the reference grid.",
    "input_schema": {
        "type": "object",
        "properties": {
            "diffs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "part": {"type": "string"},
                        "issue": {"type": "string"},
                        "axis": {"type": "string", "enum": ["x", "y", "z", "scale", "missing", "extra"]},
                        "suggested_delta": {"type": "string"},
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["part", "issue", "axis", "suggested_delta", "severity"],
                },
            },
        },
        "required": ["diffs"],
    },
}


def gen_critique(
    ref_grid_path: Path,
    render_grid_path: Path,
    ref_sil_path: Path,
    render_sil_path: Path,
    spec: dict,
    critique_model: str,
) -> tuple[list, dict]:
    """Compare reference and render (RGB + silhouette) grids and produce diffs.

    The critic sees four grids — reference photo, reference silhouette, render
    RGB, render silhouette — so it can judge shape (silhouettes) and part
    structure (RGB) together while ignoring color/texture/lighting.

    Dispatches on critique_model: "claude-*" routes to Anthropic, anything
    else routes to google-genai (Gemini). Both paths return the same shape.
    """
    args = (ref_grid_path, render_grid_path, ref_sil_path, render_sil_path, spec)
    if _is_claude_model(critique_model):
        critique, usage = _gen_critique_claude(*args, critique_model)
    else:
        critique, usage = _gen_critique_gemini(*args, critique_model)

    if not isinstance(critique, list):
        raise ValueError(f"Expected a JSON list from gen_critique, got: {type(critique)}")

    return critique, usage


def _gen_critique_gemini(
    ref_grid_path: Path,
    render_grid_path: Path,
    ref_sil_path: Path,
    render_sil_path: Path,
    spec: dict,
    gemini_model: str,
) -> tuple[list, dict]:
    def part(path):
        data, media_type = _read_image_bytes(Path(path))
        return genai_types.Part.from_bytes(data=data, mime_type=media_type)

    prompt = _CRITIQUE_PROMPT_TEMPLATE.format(spec_json=json.dumps(spec, indent=2))
    # Text labels interleaved with each image so the model cannot confuse them.
    parts = [
        prompt,
        "REFERENCE PHOTO:", part(ref_grid_path),
        "REFERENCE SILHOUETTE:", part(ref_sil_path),
        "RENDER:", part(render_grid_path),
        "RENDER SILHOUETTE:", part(render_sil_path),
    ]

    return _call_gemini_json(gemini_model, parts, _JSON_RETRY_REMINDER)


def _gen_critique_claude(
    ref_grid_path: Path,
    render_grid_path: Path,
    ref_sil_path: Path,
    render_sil_path: Path,
    spec: dict,
    claude_model: str,
) -> tuple[list, dict]:
    def image_block(path):
        b64, media_type = _read_image_b64(Path(path))
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}

    prompt = _CRITIQUE_PROMPT_TEMPLATE.format(spec_json=json.dumps(spec, indent=2))

    # Explicit text labels immediately before each image — without them the
    # model can confuse which grid is which, producing an inverted critique,
    # the worst failure mode available here (the loop "corrects" the model
    # away from the reference instead of toward it).
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "REFERENCE PHOTO:"},
        image_block(ref_grid_path),
        {"type": "text", "text": "REFERENCE SILHOUETTE:"},
        image_block(ref_sil_path),
        {"type": "text", "text": "RENDER:"},
        image_block(render_grid_path),
        {"type": "text", "text": "RENDER SILHOUETTE:"},
        image_block(render_sil_path),
    ]

    system_prompt = (
        "You compare a procedurally generated 3D model against a reference and report ONLY "
        "shape, proportion, and placement discrepancies via the emit_critique tool. The render "
        "is untextured gray geometry; never comment on color, material, texture, or lighting."
    )
    tool_input, usage = _call_claude_tool(claude_model, system_prompt, user_content, _EMIT_CRITIQUE_TOOL)
    # Unwrap the object wrapper so gen_critique returns a bare list of diffs.
    return tool_input.get("diffs", []), usage


# --------------------------------------------------------------------------
# 4) gen_asset_critique — the critic for text/image-to-asset generation
#
# Distinct from gen_critique above: there is no reference grid to diff against,
# so the critic judges the render against the SPEC'S INTENT (part counts,
# proportions, silhouette, whether it reads as the target object). It emits the
# same diff schema (_EMIT_CRITIQUE_TOOL), so gen_script consumes it unchanged.
# gen_critique is left untouched for the benchmark.
# --------------------------------------------------------------------------

_ASSET_CRITIQUE_PROMPT_TEMPLATE = """You are reviewing a STYLIZED LOW-POLY GAME ASSET against \
the art director's spec that it was built from.

The RENDER below is a grid of SIX views of the SAME model, laid out as 3 columns x 2 rows:
  top row,    left to right: FRONT, BACK, LEFT
  bottom row, left to right: RIGHT, TOP (looking straight down), THREE-QUARTER
{style_note}
READING THE GRID — DO THIS BEFORE JUDGING ANYTHING:
Look at all six views first. Judge each property ONLY from the view where that property is \
clearest and least occluded:
- HEIGHT / LENGTH / CURVE / TAPER of an upright part (trunk, stem, post, body): judge from a \
SIDE view (LEFT or RIGHT), where the part is seen edge-on and is not hidden behind anything. \
Never judge height from the TOP view — height is foreshortened to nothing there.
- COUNT and RADIAL SPACING of repeated parts (fronds, leaves, branches, legs, spokes): judge \
from the TOP view, where they fan out and can actually be counted one by one. In the FRONT \
and SIDE views repeated parts overlap and hide each other, so a count made there is worthless.
- OVERALL SILHOUETTE and whether the asset reads at a glance: judge from the FRONT view.
- DEPTH and how parts sit together in 3D: use the THREE-QUARTER view.
- Asymmetry, or a part missing from one side: compare FRONT against BACK.

OCCLUSION IS NOT A DEFECT — THIS IS THE MOST IMPORTANT RULE:
A part that looks hidden, clipped, or short IN ONE VIEW is almost always the camera angle, \
not the geometry. Fronds covering the trunk in the FRONT view do NOT mean the trunk is too \
short — look at a SIDE view, where the trunk is unobstructed, before saying anything about \
its height. A part you cannot see in one view is not missing; find it in the other five.

Before you emit ANY diff, verify the problem is visible in AT LEAST TWO views, or in the one \
view that is definitive for that property (TOP for counts, SIDE for heights). If a part looks \
wrong in one view but fine in the others, it is FINE — emit nothing for it. Reporting an \
occlusion artifact as a geometry error makes the coder break working geometry, which is far \
worse than staying silent.

The asset is meant to realize this spec:
{spec_json}

Judge the render against the SPEC'S INTENT, each from its clearest view:
- PART COUNTS (count in the TOP view): the spec gives an explicit integer count for each \
part. If the spec says 9 fronds and you count 5 IN THE TOP VIEW, that is a "missing" diff. \
Never base a count on the front or side views.
- PROPORTIONS (judge in a SIDE view): do the parts match the spec's stated ratios and sizes \
relative to each other, measured where they are unobstructed?
- SILHOUETTE (judge in the FRONT view): does the shape read cleanly and unambiguously as the \
target object at a glance? A muddy or unreadable silhouette is the most severe defect for a \
game asset.
- PLACEMENT & CONNECTION (judge in SIDE and THREE-QUARTER views): are parts positioned as the \
spec describes, and do connected parts actually meet/overlap rather than float apart or leave \
gaps? Parts that merely appear to touch in one view may be far apart — confirm in a second view.
- Does the whole thing read as "{object_name}"?

This is a STYLIZED asset, not a photo reconstruction — do not ask for realism or fine \
detail. Judge only what the spec asked for.

Return JSON ONLY — no prose, no markdown fences — as a list of objects with this schema:

[
  {{"part": str,
    "issue": str,
    "axis": "x"|"y"|"z"|"scale"|"missing"|"extra",
    "suggested_delta": str,
    "severity": "high"|"medium"|"low"}}
]

Rules:
- "part" should reference a part name from the spec when possible, or a short descriptive \
label if the issue doesn't map to a named part.
- "axis" categorizes the kind of error: a positional/rotational error along a specific axis, \
a scale error, a part the spec requires but the render is missing (or has too few of), or an \
extra part the render has that the spec does not call for.
- "suggested_delta" is a short, concrete, actionable instruction (e.g. "add 4 more fronds to \
reach the spec's 9", "move +0.3 along z", "increase radius by ~20%").
- "issue" must name the view(s) the problem is visible in (e.g. "only 5 fronds visible in \
the top view, spec calls for 9"). If you cannot name the view that proves it, do not emit it.
- NEVER emit a diff whose only evidence is a single view, unless that view is the definitive \
one for the property (TOP for counts, SIDE for heights/lengths). Occlusion artifacts are not \
defects.
- Return an empty list [] if the asset realizes the spec well — do not invent issues to fill \
the list. An empty list ends the refinement loop, which is the correct outcome when the \
asset is good.
- Do not use free-text paragraphs; every entry must fit the schema exactly.
"""

_ASSET_CRITIQUE_STYLE_NOTE = """
A STYLE REFERENCE image is also attached. It shows the intended style and subject. Compare \
the render's shape language and proportions against it — but remember it is INSPIRATION, not \
a reconstruction target, so never ask for photo-accuracy or detail it cannot express as \
low-poly geometry.
"""


def gen_asset_critique(
    render_grid_path: Path,
    spec: dict,
    critique_model: str,
    style_image: Optional[Path] = None,
) -> tuple[list, dict]:
    """Critique a rendered asset against the spec's intent; return structured diffs.

    Unlike gen_critique (which diffs a render against a reference grid), this
    judges the render against the build spec: part counts, proportions,
    silhouette readability, placement/connection. If style_image is given it is
    passed too, as inspiration to compare against — never a reconstruction target.

    Dispatches on critique_model: "claude-*" -> Anthropic, else google-genai.
    Returns (diffs, usage) with the same diff schema as the benchmark's critic.
    """
    prompt = _ASSET_CRITIQUE_PROMPT_TEMPLATE.format(
        spec_json=json.dumps(spec, indent=2),
        object_name=spec.get("object", "the target object"),
        style_note=_ASSET_CRITIQUE_STYLE_NOTE if style_image is not None else "",
    )

    if _is_claude_model(critique_model):
        critique, usage = _gen_asset_critique_claude(prompt, render_grid_path, style_image, critique_model)
    else:
        critique, usage = _gen_asset_critique_gemini(prompt, render_grid_path, style_image, critique_model)

    if not isinstance(critique, list):
        raise ValueError(f"Expected a JSON list from gen_asset_critique, got: {type(critique)}")
    return critique, usage


def _gen_asset_critique_gemini(
    prompt: str, render_grid_path: Path, style_image: Optional[Path], gemini_model: str
) -> tuple[list, dict]:
    def part(path):
        data, media_type = _read_image_bytes(Path(path))
        return genai_types.Part.from_bytes(data=data, mime_type=media_type)

    parts: list = [prompt, "RENDER:", part(render_grid_path)]
    if style_image is not None:
        parts += ["STYLE REFERENCE:", part(style_image)]
    return _call_gemini_json(gemini_model, parts, _JSON_RETRY_REMINDER)


def _gen_asset_critique_claude(
    prompt: str, render_grid_path: Path, style_image: Optional[Path], claude_model: str
) -> tuple[list, dict]:
    def image_block(path):
        b64, media_type = _read_image_b64(Path(path))
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}

    # Label each image so the critic cannot confuse the render with the style ref.
    user_content: list = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "RENDER:"},
        image_block(render_grid_path),
    ]
    if style_image is not None:
        user_content.append({"type": "text", "text": "STYLE REFERENCE:"})
        user_content.append(image_block(style_image))

    system_prompt = (
        "You review stylized low-poly game assets against the art director's spec and report "
        "shape, proportion, part-count, and placement discrepancies via the emit_critique tool."
    )
    tool_input, usage = _call_claude_tool(claude_model, system_prompt, user_content, _EMIT_CRITIQUE_TOOL)
    return tool_input.get("diffs", []), usage
