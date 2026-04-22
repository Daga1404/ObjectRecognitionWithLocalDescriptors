"""
Reconocimiento de señalamientos viales EN VIVO (RTSP) con SIFT / ORB
=====================================================================
Versión live del clasificador: recibe un stream RTSP, extrae descriptores
de cada frame y los compara contra 6 señales de referencia, dibujando
bounding box + etiqueta sobre cada detección.

Señales de referencia (archivos en sign_images/):
  sign0.jpeg → restricted_area   (Área Restringida)
  sign1.jpeg → pedestrian_zone   (Zona Peatonal)
  sign2.jpeg → robots_only       (Zona Sólo Robots — AGV triangular)
  sign3.jpeg → stop              (STOP)
  sign4.jpeg → loading_zone      (Caution / AGV Loading Zone)
  sign5.jpeg → parking_zone      (AVG Parking)

CONTROLES EN VIVO:
  · Q     → salir
  · D     → alternar detector SIFT ↔ ORB
  · S     → guardar captura PNG del frame actual
  · P     → pausa / reanudar
"""

import os
import time
import threading
import cv2
import numpy as np


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================
RTSP_URL       = "rtsp://10.43.53.23:8554/stream"
FRAME_SIZE     = (640, 480)              # (W, H) a los que se redimensiona
RECONNECT      = True

REFERENCES_DIR = "sign_images"
SCREENSHOT_DIR = "screenshots"

# Detector inicial (se puede cambiar en caliente con la tecla D)
DEFAULT_DETECTOR = "sift"                # "sift" o "orb"

# — SIFT —
SIFT_N_FEATURES      = 0
SIFT_CONTRAST_THRESH = 0.04
SIFT_EDGE_THRESH     = 10

# — ORB —
ORB_N_FEATURES   = 1500
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS     = 8

# — Matching / decisión —
LOWE_RATIO           = 0.75
MIN_GOOD_MATCHES     = 12
MIN_INLIERS          = 8
RANSAC_REPROJ_THRESH = 5.0

# — Validación del bounding box —
MIN_BBOX_AREA_FRAC = 0.002
MAX_BBOX_AREA_FRAC = 0.40
MIN_ASPECT_RATIO   = 0.30
MAX_ASPECT_RATIO   = 3.30

APPLY_CLAHE = True

# — Fallback color+forma (para señales sin features distintivos, p.ej. 'restricted_area') —
RED_FALLBACK_ENABLED  = True
RED_FALLBACK_LABEL    = "restricted_area"
RED_HSV_LOW1          = (  0,  80,  60)
RED_HSV_HIGH1         = ( 10, 255, 255)
RED_HSV_LOW2          = (170,  80,  60)
RED_HSV_HIGH2         = (179, 255, 255)
RED_MIN_AREA_FRAC     = 0.005
RED_MAX_AREA_FRAC     = 0.40
RED_MIN_CIRCULARITY   = 0.55           # 4πA/P² (anillo cerrado da valor alto tras MORPH_CLOSE)
RED_MIN_ASPECT        = 0.70
RED_MAX_ASPECT        = 1.45
RED_REQUIRE_DIAGONAL  = True           # exige una línea ~diagonal dentro del círculo

# Señales: clave → (archivo, texto en pantalla, color BGR)
SIGNS = {
    "restricted_area": ("sign0.jpeg", "AREA RESTRINGIDA",  ( 50,  50, 220)),
    "pedestrian_zone": ("sign1.jpeg", "ZONA PEATONAL",     (230, 130,  20)),
    "robots_only":     ("sign2.jpeg", "SOLO ROBOTS",       ( 30, 210, 230)),
    "stop":            ("sign3.jpeg", "STOP",              ( 30,  30, 220)),
    "loading_zone":    ("sign4.jpeg", "ZONA DE CARGA",     ( 30, 150, 255)),
    "parking_zone":    ("sign5.jpeg", "ESTACIONAMIENTO",   ( 40, 200,  40)),
}

WINDOW_NAME = "Reconocimiento de senalamientos"


# ===========================================================================
# RTSP STREAM  (misma clase que tu código de detección de formas)
# ===========================================================================
class RTSPStream:
    def __init__(self, url: str, reconnect: bool = True):
        self.url       = url
        self.reconnect = reconnect
        self._cap      = None
        self._frame    = None
        self._running  = False
        self._lock     = threading.Lock()
        self._thread   = None

    def _open_capture(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError(f"No se pudo conectar al stream RTSP: {self.url}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        return cap

    def start(self):
        self._cap     = self._open_capture()
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if self.reconnect and self._running:
                    print("[INFO] Stream perdido, reconectando...")
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    time.sleep(1)
                    try:
                        self._cap = self._open_capture()
                    except ConnectionError:
                        continue
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
        return cv2.resize(frame, FRAME_SIZE)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ===========================================================================
# DETECTORES Y MATCHERS
# ===========================================================================
def create_detector(kind):
    if kind == "sift":
        return cv2.SIFT_create(
            nfeatures         = SIFT_N_FEATURES,
            contrastThreshold = SIFT_CONTRAST_THRESH,
            edgeThreshold     = SIFT_EDGE_THRESH,
        )
    if kind == "orb":
        return cv2.ORB_create(
            nfeatures   = ORB_N_FEATURES,
            scaleFactor = ORB_SCALE_FACTOR,
            nlevels     = ORB_N_LEVELS,
        )
    raise ValueError(f"Detector desconocido: {kind}")


def create_matcher(kind):
    if kind == "sift":
        index_params  = dict(algorithm=1, trees=5)     # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    if kind == "orb":
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    raise ValueError(f"Matcher desconocido: {kind}")


# ===========================================================================
# PREPROCESAMIENTO Y CARGA DE REFERENCIAS
# ===========================================================================
def preprocess_gray(bgr, clahe):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if clahe is not None:
        gray = clahe.apply(gray)
    return gray


def load_references(detector, references_dir, clahe, kind_label=""):
    refs = {}
    for label, (filename, text, color) in SIGNS.items():
        path = os.path.join(references_dir, filename)
        if not os.path.isfile(path):
            print(f"    [AVISO] Referencia no encontrada: {path}")
            continue

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"    [AVISO] No se pudo leer: {path}")
            continue

        gray    = preprocess_gray(img, clahe)
        kp, des = detector.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            print(f"    [AVISO] Pocos keypoints en '{label}' "
                  f"({0 if des is None else len(kp)}). Se omite.")
            continue

        refs[label] = {"gray": gray, "kp": kp, "des": des,
                       "text": text, "color": color}
        print(f"    ✓ [{kind_label:4s}] {label:17s} → {len(kp):4d} keypoints")

    return refs


# ===========================================================================
# VALIDACIÓN Y DETECCIÓN
# ===========================================================================
def is_valid_box(corners, frame_shape):
    if corners is None or len(corners) != 4:
        return False
    pts_i = corners.reshape(4, 2).astype(np.int32)

    if not cv2.isContourConvex(pts_i):
        return False

    area       = cv2.contourArea(pts_i)
    h, w       = frame_shape[:2]
    frame_area = h * w
    if area < MIN_BBOX_AREA_FRAC * frame_area:
        return False
    if area > MAX_BBOX_AREA_FRAC * frame_area:
        return False

    _, _, bw, bh = cv2.boundingRect(pts_i)
    if bh == 0:
        return False
    aspect = bw / bh
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False
    return True


def detect_sign(ref, scene_kp, scene_des, matcher, frame_shape):
    if scene_des is None or len(scene_des) < 2:
        return None, 0

    try:
        knn = matcher.knnMatch(ref["des"], scene_des, k=2)
    except cv2.error:
        return None, 0

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None, len(good)

    src_pts = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_kp[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None or mask is None:
        return None, 0

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        return None, n_inliers

    h_ref, w_ref  = ref["gray"].shape[:2]
    ref_corners   = np.float32([[0, 0], [0, h_ref-1],
                                [w_ref-1, h_ref-1], [w_ref-1, 0]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(ref_corners, H)

    if not is_valid_box(scene_corners, frame_shape):
        return None, n_inliers
    return scene_corners, n_inliers


# ===========================================================================
# DIBUJO
# ===========================================================================
def draw_detection(frame, corners, text, color, n_inliers):
    pts = corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(frame, [pts], True, color, 3, cv2.LINE_AA)

    x, y, _, _  = cv2.boundingRect(pts)
    label       = f"{text} ({n_inliers})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_top       = max(0, y - th - 10)
    cv2.rectangle(frame, (x, y_top), (x + tw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, detector_name, fps, n_hits, paused):
    line1 = f"Detector: {detector_name.upper()}   FPS: {fps:5.1f}   Hits: {n_hits}"
    line2 = "Q=salir   D=alternar detector   S=captura   P=pausa"
    if paused:
        line2 = "[PAUSA]   " + line2

    (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    box_w = max(tw1, tw2) + 14
    box_h = th1 + th2 + 22

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, line1, (7, th1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (7, th1 + th2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ===========================================================================
# FALLBACK COLOR + FORMA  (anillo rojo con barra diagonal)
# ===========================================================================
def _has_diagonal_line(roi_mask):
    h, w = roi_mask.shape[:2]
    if h < 10 or w < 10:
        return False
    edges = cv2.Canny(roi_mask, 50, 150)
    min_len = int(0.4 * min(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return False
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        ang = abs(np.degrees(np.arctan2(dy, dx)))   # 0..180
        if 20 <= ang <= 70 or 110 <= ang <= 160:
            return True
    return False


def detect_red_prohibition(frame):
    """Devuelve corners (4x1x2 float32) del mejor candidato a anillo rojo, o None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(RED_HSV_LOW1,  np.uint8),
                          np.array(RED_HSV_HIGH1, np.uint8))
    m2 = cv2.inRange(hsv, np.array(RED_HSV_LOW2,  np.uint8),
                          np.array(RED_HSV_HIGH2, np.uint8))
    raw = cv2.bitwise_or(m1, m2)

    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=2)

    h, w        = frame.shape[:2]
    frame_area  = h * w

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_corners = None
    best_score   = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < RED_MIN_AREA_FRAC * frame_area:
            continue
        if area > RED_MAX_AREA_FRAC * frame_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < RED_MIN_CIRCULARITY:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if aspect < RED_MIN_ASPECT or aspect > RED_MAX_ASPECT:
            continue

        if RED_REQUIRE_DIAGONAL and not _has_diagonal_line(raw[y:y+bh, x:x+bw]):
            continue

        score = circularity * area
        if score > best_score:
            best_score = score
            best_corners = np.float32([
                [x,        y       ],
                [x,        y + bh  ],
                [x + bw,   y + bh  ],
                [x + bw,   y       ],
            ]).reshape(-1, 1, 2)

    return best_corners


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    if not os.path.isdir(REFERENCES_DIR):
        print(f"[ERROR] Carpeta de referencias no encontrada: '{REFERENCES_DIR}'")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if APPLY_CLAHE else None

    # Precargar detectores + referencias para AMBOS (permite alternar en caliente)
    detectors_cache = {}
    for kind in ("sift", "orb"):
        print(f"[INFO] Inicializando {kind.upper()} y cargando referencias...")
        det  = create_detector(kind)
        mat  = create_matcher(kind)
        refs = load_references(det, REFERENCES_DIR, clahe, kind_label=kind)
        detectors_cache[kind] = (det, mat, refs)
        if not refs:
            print(f"  [AVISO] No se cargaron referencias para {kind.upper()}.")

    # Asegurar que al menos el detector inicial tenga referencias
    current_kind = DEFAULT_DETECTOR
    if not detectors_cache[current_kind][2]:
        alt = "orb" if current_kind == "sift" else "sift"
        if detectors_cache[alt][2]:
            print(f"[INFO] {current_kind.upper()} sin referencias; usando {alt.upper()}.")
            current_kind = alt
        else:
            print("[ERROR] Ningún detector tiene referencias cargadas. Aborta.")
            return

    # Contadores acumulados por detector
    totals = {kind: {label: 0 for label in SIGNS} for kind in ("sift", "orb")}

    print(f"\n[INFO] Conectando a {RTSP_URL}")
    print(f"[INFO] Detector inicial: {current_kind.upper()}")
    print("[INFO] Controles:  Q=salir  D=alternar detector  S=captura  P=pausa\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused     = False
    last_frame = None
    frame_idx  = 0
    last_fps_t = time.time()
    last_fps_n = 0
    cur_fps    = 0.0

    try:
        with RTSPStream(RTSP_URL, reconnect=RECONNECT) as stream:
            while True:
                if not paused:
                    frame = stream.read()
                    if frame is None:
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                        continue

                    det, mat, refs = detectors_cache[current_kind]

                    gray                = preprocess_gray(frame, clahe)
                    scene_kp, scene_des = det.detectAndCompute(gray, None)

                    hits = 0
                    restricted_hit = False
                    for label, ref in refs.items():
                        corners, n_in = detect_sign(ref, scene_kp, scene_des,
                                                    mat, frame.shape)
                        if corners is not None:
                            draw_detection(frame, corners, ref["text"],
                                           ref["color"], n_in)
                            totals[current_kind][label] += 1
                            hits += 1
                            if label == RED_FALLBACK_LABEL:
                                restricted_hit = True

                    if (RED_FALLBACK_ENABLED
                            and not restricted_hit
                            and RED_FALLBACK_LABEL in SIGNS):
                        fb_corners = detect_red_prohibition(frame)
                        if fb_corners is not None:
                            _, fb_text, fb_color = SIGNS[RED_FALLBACK_LABEL]
                            draw_detection(frame, fb_corners,
                                           fb_text + " [color]",
                                           fb_color, 0)
                            totals[current_kind][RED_FALLBACK_LABEL] += 1
                            hits += 1

                    now = time.time()
                    if now - last_fps_t >= 0.5:
                        cur_fps    = (frame_idx - last_fps_n) / (now - last_fps_t)
                        last_fps_t = now
                        last_fps_n = frame_idx

                    draw_hud(frame, current_kind, cur_fps, hits, paused=False)
                    last_frame = frame
                    frame_idx += 1
                else:
                    if last_frame is None:
                        continue
                    frame = last_frame.copy()
                    draw_hud(frame, current_kind, cur_fps, 0, paused=True)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    new_kind = "orb" if current_kind == "sift" else "sift"
                    if detectors_cache[new_kind][2]:
                        current_kind = new_kind
                        print(f"[INFO] Detector → {current_kind.upper()}")
                    else:
                        print(f"[AVISO] {new_kind.upper()} no tiene referencias cargadas.")
                elif key == ord('s'):
                    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
                    path = os.path.join(
                        SCREENSHOT_DIR,
                        f"capture_{current_kind}_{int(time.time())}.png"
                    )
                    cv2.imwrite(path, frame)
                    print(f"[INFO] Captura guardada: {path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"[INFO] {'PAUSA' if paused else 'REANUDADO'}")

    except ConnectionError as e:
        print(f"[ERROR] {e}")

    cv2.destroyAllWindows()

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN ACUMULADO (frames con detección)")
    print("=" * 60)
    for kind in ("sift", "orb"):
        print(f"\n  {kind.upper()}:")
        for label, count in sorted(totals[kind].items(), key=lambda kv: -kv[1]):
            print(f"    {label:18s}: {count:5d}")


if __name__ == "__main__":
    main()
