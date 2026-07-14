# Image to 3D (Blender + LLM)

Generates a Blender geometry script from a reference image and exports a GLB.

Output is primitive-assembly geometry: blocky, prototype-grade, parts may not
fully connect. Suitable for greyboxing, not finished assets.

## Setup
    pip install anthropic google-genai lpips torch pillow scikit-image rembg "rembg[cpu]" scipy
    export ANTHROPIC_API_KEY=...        # Windows: set ANTHROPIC_API_KEY=...

Install Blender 4.2+. If it isn't on PATH, set BLENDER_BIN to the blender binary.

## Generate a model
    python generate.py --image chair.jpg --out chair.glb --iters 3

Opens in any glTF viewer or Three.js GLTFLoader.

## Benchmark (optional — compares models)
    python benchmark.py --ref chair.jpg --max-iters 5
