# Grip Studio

Local browser tool for personalizing a scanned prybar grip from three hand measurements. Python handles mesh deformation and integrity checks; bundled Three.js displays the grip and an illustrative hand. No upload is required at runtime.

## Run locally

Tested with Python 3.12 on macOS Apple Silicon. PyMeshLab requires compatible native wheels.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/configure_baseline.py
.venv/bin/python app.py
```

Open http://127.0.0.1:8765. On Windows substitute `.venv\Scripts\python` for `.venv/bin/python`.

The first-time configuration requires the hand measurements of the person who formed the supplied scan. Local calibration is intentionally excluded from the repository. The supplied model is in millimetres, with thumb end A and the current saved design orientation. Configuration asks for acceptance of the repair report and landmarks; it never invents reference measurements.

## Use

Enter hand length, hand breadth, and palm length in millimetres, generate a preview, and download the validated prototype pack. The pack includes STL, profile and validation results. The public interface exposes only the three measurement inputs. Orbit and zoom the preview with the mouse.

Hand length is wrist crease to middle fingertip. Palm length uses the same wrist crease to middle-finger base crease. Breadth is across the knuckles excluding the thumb. Do not mix landmark definitions with ANSUR measurements without checking their compatibility.

## Deformation

Hand breadth controls axial grip length. Hand length minus palm length controls cross-sectional scaling. Palm length adds a blended palm-side correction. The original imprint is smoothly deformed rather than replaced with new finger grooves. All model features resize, including any mating features. These are experimental engineering rules, not validated ANSUR ergonomic formulas.

The hand is a procedural illustration, not a hand scan or contact simulation, and is excluded from STL exports. Export checks cover closure, normals, degenerate faces and detected self-intersections. Physical comfort and load-bearing suitability require separate testing.

## Tests

```sh
.venv/bin/python -m pytest -q
node tests/hand-position.mjs
```

`scripts/test_hand_sizes.py` downloads public ANSUR 1988 data and generates five size tests. Palm lengths in those tests are estimated from the configured reference ratio. `scripts/render_hand.py` produces an offline illustration and currently uses a macOS font path.

## Contents and third-party notices

- `grip.py`: mesh preparation, deformation and validation
- `hand.py`: illustrative hand geometry
- `app.py`: localhost server
- `static/`: browser interface and bundled Three.js
- `data/smoothed_mergedhandle_V2.stl`: supplied source grip
- `tests/`: regression tests

Three.js license is included at `static/vendor/THREE-LICENSE.txt`. Python dependencies retain their respective licenses. Personal photos, local reference measurements, exports, and virtual environments are not included.
