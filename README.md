# Object Recognition With Local Descriptors

Two-script pipeline for recognizing six traffic / facility signs using **local feature descriptors** (SIFT and ORB), with a per-sign **color+shape fallback** for icons that are too feature-poor for descriptor matching.

- **`part1.py`** — Offline batch evaluator. Runs **both SIFT and ORB** over every test image and writes side-by-side comparisons + a global summary. Used to validate the pipeline and tune parameters before going live.
- **`part2.py`** — Live RTSP recognizer. Pulls frames from an IP camera, runs the chosen detector in real time, supports **hot-swapping SIFT ↔ ORB** while running, and falls back to a color+shape detector for the prohibition sign (which has no usable local features).

---

## Table of contents

1. [The six reference signs](#the-six-reference-signs)
2. [Repository layout](#repository-layout)
3. [Requirements & installation](#requirements--installation)
4. [Algorithmic background](#algorithmic-background)
5. [`part1.py` — offline batch evaluator](#part1py--offline-batch-evaluator)
6. [`part2.py` — live RTSP recognizer](#part2py--live-rtsp-recognizer)
7. [The color+shape fallback (`detect_red_prohibition`)](#the-colorshape-fallback-detect_red_prohibition)
8. [Tuning guide](#tuning-guide)
9. [Known limitations](#known-limitations)
10. [Troubleshooting](#troubleshooting)

---

## The six reference signs

All references live in `sign_images/`. Both scripts read this directory; the filename → semantic-label mapping is hard-coded in the `SIGNS` dict at the top of each script.

| Key in `SIGNS`     | File         | On-screen label    | Color (BGR)        | Notes                                                       |
|--------------------|--------------|--------------------|--------------------|-------------------------------------------------------------|
| `restricted_area`  | `sign0.jpeg` | AREA RESTRINGIDA   | `( 50,  50, 220)`  | Red ring + diagonal slash. Feature-poor → uses fallback.    |
| `pedestrian_zone`  | `sign1.jpeg` | ZONA PEATONAL      | `(230, 130,  20)`  | Pictogram with figure.                                      |
| `robots_only`      | `sign2.jpeg` | SOLO ROBOTS        | `( 30, 210, 230)`  | Triangular AGV-only sign.                                   |
| `stop`             | `sign3.jpeg` | STOP               | `( 30,  30, 220)`  | Octagon, large bold text — feature-rich.                    |
| `loading_zone`     | `sign4.jpeg` | ZONA DE CARGA      | `( 30, 150, 255)`  | "CAUTION / AGV LOADING ZONE" with forklift icon.            |
| `parking_zone`     | `sign5.jpeg` | ESTACIONAMIENTO    | `( 40, 200,  40)`  | "AVG PARKING".                                              |

To change which file maps to which label, edit the `SIGNS` dictionary at the top of `part1.py` / `part2.py` — both scripts share the same structure.

---

## Repository layout

```
ObjectRecognitionWithLocalDescriptors/
├── part1.py            # Offline batch: SIFT vs ORB on static images
├── part2.py            # Live RTSP recognizer + color+shape fallback
├── sign_images/        # The 6 reference templates (sign0..sign5.jpeg)
├── outputs/            # Auto-created by part1.py: per-image PNGs + comparisons
├── screenshots/        # Auto-created by part2.py when you press 'S'
└── README.md
```

---

## Requirements & installation

- **Python 3.8+**
- **OpenCV with `xfeatures2d`** (SIFT is in mainline `opencv-python` since 4.4; if SIFT fails to instantiate, install `opencv-contrib-python` instead).
- **NumPy**
- **FFmpeg** must be available to OpenCV for RTSP capture in `part2.py` (`opencv-python` ships with it on Windows wheels; on Linux you may need a system FFmpeg).

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
pip install opencv-contrib-python numpy
```

If you only run `part1.py`, plain `opencv-python` is enough on most platforms.

---

## Algorithmic background

Both scripts share the same recognition pipeline. Understanding each stage makes the configuration knobs make sense.

### 1. Grayscale + CLAHE preprocessing
`preprocess_gray()` converts BGR → grayscale, then optionally applies **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) with `clipLimit=2.0`, `tileGridSize=(8,8)`. CLAHE boosts local contrast without blowing out bright regions — important for printed signs photographed indoors where lighting is uneven.

Toggle with `APPLY_CLAHE` in either script.

### 2. Keypoint detection + descriptor extraction
Two interchangeable feature detectors:

- **SIFT** (`cv2.SIFT_create`) — scale + rotation invariant, 128-D float descriptors, very robust but slower. Tuned via `SIFT_N_FEATURES`, `SIFT_CONTRAST_THRESH`, `SIFT_EDGE_THRESH`.
- **ORB** (`cv2.ORB_create`) — binary 256-bit descriptors, much faster, slightly less robust to viewpoint. Tuned via `ORB_N_FEATURES`, `ORB_SCALE_FACTOR`, `ORB_N_LEVELS`.

Both produce `(keypoints, descriptors)` for the reference image (computed once at startup) and for every scene image / live frame.

### 3. Descriptor matching
Each detector pairs with the appropriate matcher (`create_matcher()`):

- **SIFT → FLANN (KD-tree)** — fast approximate nearest-neighbor search over float descriptors.
- **ORB → BFMatcher with `NORM_HAMMING`** — exact Hamming-distance matching for binary descriptors.

For each query keypoint we ask for its **two** nearest neighbors via `knnMatch(..., k=2)`.

### 4. Lowe's ratio test
A match `m` is kept only if its distance is much smaller than the second-best `n`:

```
m.distance < LOWE_RATIO * n.distance       # default LOWE_RATIO = 0.75
```

This filters out keypoints in repetitive textures (where the best and 2nd-best are nearly tied).

### 5. RANSAC homography
The surviving "good" matches are fed into `cv2.findHomography(..., cv2.RANSAC, RANSAC_REPROJ_THRESH)`. RANSAC iteratively picks 4-point subsets, fits a homography, and counts which other matches agree (inliers) within `RANSAC_REPROJ_THRESH` pixels. The largest consensus set wins.

A detection is accepted only if:

- `len(good) >= MIN_GOOD_MATCHES` (filtered raw matches)
- `n_inliers >= MIN_INLIERS` (after RANSAC)

### 6. Bounding-box validation (`is_valid_box`)
The reference rectangle is projected through the homography to get four corner points in the scene. We then sanity-check the resulting quadrilateral:

| Check                            | Why                                                           |
|----------------------------------|---------------------------------------------------------------|
| `cv2.isContourConvex`            | Reject self-intersecting / folded-over quads from bad H.      |
| `MIN_BBOX_AREA_FRAC` ≤ area ≤ `MAX_BBOX_AREA_FRAC` × frame_area | Reject postage-stamp false positives and "fills the screen" hallucinations. |
| `MIN_ASPECT_RATIO ≤ w/h ≤ MAX_ASPECT_RATIO` | Reject extremely stretched matches (degenerate H). |

Only quads passing all three are drawn.

### 7. Drawing
`draw_detection()` overlays a thick polyline along the projected corners and a colored label box with `"<text> (<n_inliers>)"`. The color comes from the third tuple field in `SIGNS`.

---

## `part1.py` — offline batch evaluator

### Purpose
Validate the recognition pipeline on a folder of static images **before** going live, and produce side-by-side **SIFT vs ORB** comparisons plus a global summary table.

### Pipeline (`main()`)
1. Verifies `REFERENCES_DIR` and `TEST_IMAGES_DIR` exist; creates `OUTPUTS_DIR` if missing.
2. Lists every image in `TEST_IMAGES_DIR` matching `IMAGE_EXTS = (.jpg, .jpeg, .png, .bmp, .webp)`.
3. For **both** SIFT and ORB:
   - Builds the detector + matcher.
   - Calls `load_references()`, which runs the detector once per template and caches `{label → {gray, kp, des, text, color}}`.
4. For each test image:
   - Reads it, resizes to `MAX_TEST_SIZE` (longest side ≤ 1280 px by default; preserves aspect).
   - Runs `process_image()` once with SIFT and once with ORB. Each call:
     - Extracts scene keypoints.
     - Iterates over every reference, calling `detect_sign()` to produce `(corners, n_good, n_inliers)`.
     - Draws accepted detections on a copy of the frame.
     - Returns the annotated frame plus a stats dict.
   - Writes three PNGs to `OUTPUTS_DIR`:
     - `<base>_sift.png`
     - `<base>_orb.png`
     - `<base>_comparison.png` — SIFT (left) and ORB (right) joined with a 4-px white separator. Heights are equalized first if they differ.
   - Updates per-detector totals.
5. Prints a global summary: total time, average ms/image, average keypoints/image, and **per-sign detection counts** as `count/N (pct%)` for each detector — the table that tells you which detector wins for which sign.

### Key config (top of file)
```python
REFERENCES_DIR  = "sign_images"
TEST_IMAGES_DIR = "sign_images"      # by default, references double as test images
OUTPUTS_DIR     = "outputs"
MAX_TEST_SIZE   = 1280               # set to None to disable resizing
APPLY_CLAHE     = True

# Higher-tolerance thresholds than part2 (offline can afford to be permissive)
ORB_N_FEATURES   = 3000
LOWE_RATIO       = 0.75
MIN_GOOD_MATCHES = 10
MIN_INLIERS      =  7
```

The defaults are deliberately **more permissive than `part2.py`** because static images don't suffer motion blur and you want to see borderline detections during evaluation.

### Usage
```bash
python part1.py
```
Drop additional test photos into `TEST_IMAGES_DIR` (or change the path) and re-run.

### Output interpretation

- **`*_comparison.png`** is the headline artifact. Each side shows the detector name, scene keypoint count, and total processing time in the top banner. Detected signs are drawn with their assigned color.
- **`(sin detección — mejor candidato: …, good=N)`** lines in the console name the closest-but-rejected reference and its raw match count — useful for figuring out why something *almost* matched.
- The final summary's per-sign percentages are the main quality signal. A sign at 0% across both detectors → fix the reference (see [Known limitations](#known-limitations)).

---

## `part2.py` — live RTSP recognizer

### Purpose
Run continuous recognition over an RTSP video stream (e.g. an IP camera or `mediamtx` server) with sub-second latency, while letting the operator hot-swap detectors and capture frames on demand.

### High-level architecture

```
                ┌──────────────────────┐
RTSP camera ──▶ │  RTSPStream thread   │  (background, daemon)
                │  cv2.VideoCapture +  │
                │  reconnect-on-loss   │
                └──────────┬───────────┘
                           │ latest frame (lock-protected)
                           ▼
                  ┌──────────────────┐
                  │  main() loop     │
                  │  preprocess →    │
                  │  detect+match →  │
                  │  fallback →      │
                  │  draw → imshow   │
                  └──────────────────┘
```

### `RTSPStream` class
A small threaded wrapper around `cv2.VideoCapture`:

- Opens the stream with `cv2.CAP_FFMPEG` and forces `FRAME_SIZE` (640×480 by default).
- A background daemon thread (`_capture_loop`) continuously calls `cap.read()` and stores the **latest** frame under a lock. Old frames are silently dropped — `read()` always returns the freshest one. This avoids the typical "RTSP buffer lag" problem where polling reads stale frames.
- On read failure with `RECONNECT=True`, sleeps 1 s and reopens the capture. Recovers automatically from temporary network drops.
- Implements `__enter__`/`__exit__` so it can be used as a `with` block — the stream is released and the thread joined cleanly on exit.

### Detector hot-swap
At startup `main()` builds **both** SIFT and ORB stacks (detector + matcher + reference descriptors) into `detectors_cache`. Pressing **D** flips `current_kind`, which simply changes which entry of the cache the loop pulls from. No re-initialization, no reference reloading — the swap is instant.

### Live controls

| Key | Action                                                       |
|-----|--------------------------------------------------------------|
| `Q` | Quit                                                         |
| `D` | Toggle SIFT ↔ ORB                                            |
| `S` | Save current frame to `screenshots/capture_<det>_<ts>.png`   |
| `P` | Pause / resume (last frame stays on screen with `[PAUSA]`)   |

### HUD overlay (`draw_hud`)
Top-left semi-transparent box shows the active detector, current FPS (computed over rolling 0.5 s windows), the number of detected signs in the current frame, and the controls reminder.

### Per-frame loop
```
1. RTSPStream.read()                   → latest BGR frame, resized to FRAME_SIZE
2. preprocess_gray + CLAHE
3. detector.detectAndCompute(...)      → scene_kp, scene_des
4. for each reference:
       detect_sign(...)                → corners or None
       if corners: draw_detection
5. if 'restricted_area' was NOT detected and RED_FALLBACK_ENABLED:
       detect_red_prohibition(frame)   → corners or None
       if corners: draw_detection with "[color]" suffix
6. draw_hud, imshow, handle key
```

### Key config (top of file)
```python
RTSP_URL         = "rtsp://10.43.53.23:8554/stream"
FRAME_SIZE       = (640, 480)
RECONNECT        = True
DEFAULT_DETECTOR = "sift"           # "sift" or "orb"

# Stricter than part1 — live needs to reject false positives aggressively
ORB_N_FEATURES   = 1500
MIN_GOOD_MATCHES = 12
MIN_INLIERS      =  8
MAX_BBOX_AREA_FRAC = 0.40           # reject "fills half the screen" matches
MIN_ASPECT_RATIO   = 0.30
MAX_ASPECT_RATIO   = 3.30
```

### Final summary
On exit (after `Q`), prints a per-detector frame-count tally per sign. Each tally is the number of **frames in which that sign was detected at least once** since the script started — useful for comparing detector recall across a recorded session.

### Usage
```bash
python part2.py
```
Edit `RTSP_URL` to match your camera. The stream URL format depends on the camera firmware (common patterns: `rtsp://<user>:<pass>@<ip>:<port>/<path>`).

---

## The color+shape fallback (`detect_red_prohibition`)

### Why it exists
The `restricted_area` reference (`sign0.jpeg`) is a generic prohibition icon: a red ring with a diagonal bar over a white background. SIFT and ORB both fail on it because:

1. **No corners.** A smooth circle has zero corners; the diagonal bar adds at most ~4 corner-like endpoints — already below `MIN_GOOD_MATCHES`.
2. **Symmetry kills the ratio test.** Every position along the ring has near-identical descriptors, so `m.distance ≈ n.distance` and Lowe's ratio test (`< 0.75 × n.distance`) rejects nearly every candidate.
3. **No internal texture or text.** Compare with `sign3.jpeg` (STOP), where the bold letters generate dozens of distinctive keypoints.

The other five signs detect reliably (see your `screenshots/` for examples — `ZONA DE CARGA` lit up with 67 inliers in the same scene where `AREA RESTRINGIDA` got zero hits). The fix is a **separate, single-purpose detector** that runs only when SIFT/ORB miss this one sign.

### Algorithm
1. **HSV color mask.** Red wraps around the H axis, so two `inRange` calls (H ∈ [0,10] and H ∈ [170,179]) are OR'd together. Saturation ≥ 80 and value ≥ 60 reject pinks and washed-out reds.
2. **Morphological close** (5×5 ellipse, 2 iterations) bridges the ring + slash into one solid blob. This makes the contour an actual disc instead of an annulus, which is much easier to filter.
3. **External contours** on the closed mask. Each candidate is scored against four filters:
   - **Area fraction**: `RED_MIN_AREA_FRAC ≤ area / frame_area ≤ RED_MAX_AREA_FRAC`
   - **Circularity**: `4πA / P² ≥ RED_MIN_CIRCULARITY` (1.0 = perfect circle)
   - **Aspect ratio**: `RED_MIN_ASPECT ≤ w/h ≤ RED_MAX_ASPECT`
   - **Diagonal-line check** (optional, `RED_REQUIRE_DIAGONAL`): runs `Canny + HoughLinesP` on the *un-closed* mask cropped to the candidate ROI, requires at least one line of length ≥ 40% of the ROI's smaller side at an angle in [20°, 70°] ∪ [110°, 160°].
4. **Best score wins** — `circularity × area`. Returns a 4-corner box in the same `(4, 1, 2) float32` layout as `detect_sign`, so it plugs straight into `draw_detection`.

### Wiring
The fallback only fires if SIFT/ORB did **not** detect `restricted_area` in the current frame. Detected boxes are labeled `"AREA RESTRINGIDA [color] (0)"` — the `[color]` tag and zero inlier count make it visually distinct from a feature-matched detection.

### Tunables (top of `part2.py`)
```python
RED_FALLBACK_ENABLED  = True
RED_FALLBACK_LABEL    = "restricted_area"
RED_HSV_LOW1, RED_HSV_HIGH1   # H ∈ [0, 10]
RED_HSV_LOW2, RED_HSV_HIGH2   # H ∈ [170, 179]
RED_MIN_AREA_FRAC     = 0.005
RED_MAX_AREA_FRAC     = 0.40
RED_MIN_CIRCULARITY   = 0.55
RED_MIN_ASPECT        = 0.70
RED_MAX_ASPECT        = 1.45
RED_REQUIRE_DIAGONAL  = True
```

---

## Tuning guide

| Symptom                                      | Knob                                        | Direction                  |
|----------------------------------------------|---------------------------------------------|----------------------------|
| Too many false positives on textured walls   | `MIN_GOOD_MATCHES`, `MIN_INLIERS`           | ↑                          |
| Sign detected but bbox wraps half the frame  | `MAX_BBOX_AREA_FRAC`, `MAX_ASPECT_RATIO`    | ↓                          |
| Real signs missed at distance                | `ORB_N_FEATURES`, `MIN_INLIERS`             | ↑ features, ↓ min inliers  |
| Repetitive-pattern signs (text, dots) ignored| `LOWE_RATIO`                                | ↑ to ~0.85 (more permissive)|
| Color fallback fires on chairs / clothing    | `RED_MIN_CIRCULARITY`, `RED_MIN_ASPECT`     | ↑ both, narrow aspect band |
| Color fallback misses sign on grey paper     | `RED_HSV_LOW1[1]`, `RED_HSV_LOW2[1]` (saturation) | ↓ to ~50              |
| RTSP feels laggy                             | `FRAME_SIZE`                                | ↓ (e.g. 480×360)           |
| Live FPS too low                             | Switch default to ORB (`DEFAULT_DETECTOR = "orb"`) | —                   |

---

## Known limitations

- **`sign0.jpeg` cannot be matched by SIFT/ORB.** This is fundamental to the algorithm class, not a bug. The color+shape fallback exists specifically to cover this case. The cheapest "real" fix is to replace the reference with one that includes text or distinctive graphics.
- **The pipeline assumes a roughly planar sign.** RANSAC homography only models projective transforms of a flat surface. Curved or folded signs will fit poorly and get rejected by `is_valid_box`.
- **No temporal smoothing.** Each live frame is independent — a flickering match (in/out across frames) is not suppressed. If you need stability, wrap detections in a short ring buffer and require ≥ K out of N frames before reporting.
- **Network-only RTSP.** No support for local video files in `part2.py`. To test offline, swap `cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)` for a file path.
- **Single-class assumption per sign.** Each reference is matched independently; there's no cross-reference suppression. In practice this is rarely an issue because RANSAC + bbox validation already kill duplicate hits.

---

## Troubleshooting

**`[ERROR] Carpeta no encontrada: 'X'`** — `part1.py` aborts if `REFERENCES_DIR` or `TEST_IMAGES_DIR` is missing. Check the paths near the top of the file; they are relative to the working directory you launched Python from, not to the script's location.

**`No se pudo conectar al stream RTSP`** — `cv2.VideoCapture` couldn't open the URL. Verify with VLC first (`vlc rtsp://...`); if VLC works but OpenCV doesn't, you likely need a build with FFmpeg (`opencv-contrib-python` on Linux, or set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` to force TCP).

**`AttributeError: module 'cv2' has no attribute 'SIFT_create'`** — install `opencv-contrib-python` (uninstall `opencv-python` first to avoid conflicts).

**Stream connects but `read()` returns `None` forever** — the background thread reconnects silently; check the console for `[INFO] Stream perdido, reconectando...`. If it loops, the camera is publishing but OpenCV can't decode the codec — try a smaller resolution on the camera side.

**Detected boxes flicker on / off rapidly** — your descriptors are at the threshold. Bump `MIN_INLIERS` by 1-2, or pre-blur the frame slightly (`cv2.GaussianBlur(frame, (3,3), 0)` before `preprocess_gray`).

**Color fallback marks the wall red** — your scene lighting is pushing reds toward orange/pink. Tighten `RED_HSV_HIGH1[0]` from 10 → 7 and `RED_HSV_LOW2[0]` from 170 → 175.
