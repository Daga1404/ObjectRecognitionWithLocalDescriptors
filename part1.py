"""
Reconocimiento de señalamientos sobre IMÁGENES ESTÁTICAS  —  SIFT vs ORB
=========================================================================
Procesa cada imagen en `signs/test_images/` con AMBOS detectores y guarda
una comparativa lado-a-lado en `signs/outputs/`. Ideal para validar el
pipeline antes de pasar a video en vivo.

Señales de referencia (archivos en sign_images/):
  sign0.jpeg → restricted_area   (Área Restringida)
  sign1.jpeg → pedestrian_zone   (Zona Peatonal)
  sign2.jpeg → robots_only       (Zona Sólo Robots — AGV triangular)
  sign3.jpeg → stop              (STOP)
  sign4.jpeg → loading_zone      (Caution / AGV Loading Zone)
  sign5.jpeg → parking_zone      (AVG Parking)

USO:
  1. Poner imágenes de prueba (fotos con las señales) en  signs/test_images/
  2. python traffic_sign_image_test.py
  3. Ver resultados en                                    signs/outputs/
     · <nombre>_sift.png      → sólo detección con SIFT
     · <nombre>_orb.png       → sólo detección con ORB
     · <nombre>_comparison.png → SIFT a la izquierda, ORB a la derecha

Al final imprime en consola un resumen con:
  · # de keypoints por detector y por imagen
  · # de matches buenos y inliers por señal detectada
  · tiempo de procesamiento SIFT vs ORB
"""

import os
import time
import cv2
import numpy as np


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================
REFERENCES_DIR  = "sign_images"
TEST_IMAGES_DIR = "sign_images"
OUTPUTS_DIR     = "outputs"

# Opcional: redimensionar imagen de prueba al procesar (None = mantener original)
MAX_TEST_SIZE = 1280                 # lado mayor máximo; None para no redimensionar

# — SIFT —
SIFT_N_FEATURES      = 0
SIFT_CONTRAST_THRESH = 0.04
SIFT_EDGE_THRESH     = 10

# — ORB —
ORB_N_FEATURES   = 3000              # más features para imagen estática (es gratis aquí)
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS     = 8

# — Matching / decisión —
LOWE_RATIO           = 0.75
MIN_GOOD_MATCHES     = 10            # un poco más permisivo que en live
MIN_INLIERS          = 7
RANSAC_REPROJ_THRESH = 5.0

# — Validación del bounding box —
MIN_BBOX_AREA_FRAC = 0.002
MAX_BBOX_AREA_FRAC = 0.60
MIN_ASPECT_RATIO   = 0.25
MAX_ASPECT_RATIO   = 4.00

APPLY_CLAHE = True

# Señales:  clave → (archivo, texto en pantalla, color BGR)
SIGNS = {
    "restricted_area": ("sign0.jpeg", "AREA RESTRINGIDA",  ( 50,  50, 220)),
    "pedestrian_zone": ("sign1.jpeg", "ZONA PEATONAL",     (230, 130,  20)),
    "robots_only":     ("sign2.jpeg", "SOLO ROBOTS",       ( 30, 210, 230)),
    "stop":            ("sign3.jpeg", "STOP",              ( 30,  30, 220)),
    "loading_zone":    ("sign4.jpeg", "ZONA DE CARGA",     ( 30, 150, 255)),
    "parking_zone":    ("sign5.jpeg", "ESTACIONAMIENTO",   ( 40, 200,  40)),
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def load_references(detector, references_dir, clahe, kind_label):
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
    """Devuelve (corners, n_good, n_inliers). corners=None si no se aceptó."""
    if scene_des is None or len(scene_des) < 2:
        return None, 0, 0

    try:
        knn = matcher.knnMatch(ref["des"], scene_des, k=2)
    except cv2.error:
        return None, 0, 0

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    n_good = len(good)
    if n_good < MIN_GOOD_MATCHES:
        return None, n_good, 0

    src_pts = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_kp[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None or mask is None:
        return None, n_good, 0

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        return None, n_good, n_inliers

    h_ref, w_ref  = ref["gray"].shape[:2]
    ref_corners   = np.float32([[0, 0], [0, h_ref-1],
                                [w_ref-1, h_ref-1], [w_ref-1, 0]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(ref_corners, H)

    if not is_valid_box(scene_corners, frame_shape):
        return None, n_good, n_inliers
    return scene_corners, n_good, n_inliers


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


def draw_header(frame, text, color=(30, 30, 30)):
    """Banda superior con el nombre del detector."""
    h, w = frame.shape[:2]
    bar_h = 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


# ===========================================================================
# PROCESAMIENTO DE UNA IMAGEN CON UN DETECTOR
# ===========================================================================
def process_image(bgr, detector, matcher, refs, clahe):
    """
    Devuelve:
      · annotated (BGR con detecciones dibujadas)
      · stats: dict con n_keypoints, tiempo y detalles por señal
    """
    annotated = bgr.copy()

    t0 = time.time()
    gray = preprocess_gray(bgr, clahe)
    scene_kp, scene_des = detector.detectAndCompute(gray, None)
    t_extract = time.time() - t0

    t0 = time.time()
    per_sign = {}
    for label, ref in refs.items():
        corners, n_good, n_in = detect_sign(ref, scene_kp, scene_des,
                                            matcher, bgr.shape)
        per_sign[label] = {
            "detected":  corners is not None,
            "n_good":    n_good,
            "n_inliers": n_in,
        }
        if corners is not None:
            draw_detection(annotated, corners, ref["text"],
                           ref["color"], n_in)
    t_match = time.time() - t0

    return annotated, {
        "n_keypoints": len(scene_kp),
        "t_extract":   t_extract,
        "t_match":     t_match,
        "per_sign":    per_sign,
    }


# ===========================================================================
# REDIMENSIONADO MANTENIENDO ASPECTO
# ===========================================================================
def resize_max(img, max_side):
    if max_side is None:
        return img
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / s
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    for d in (REFERENCES_DIR, TEST_IMAGES_DIR):
        if not os.path.isdir(d):
            print(f"[ERROR] Carpeta no encontrada: '{d}'")
            return
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    test_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not test_files:
        print(f"[ERROR] No hay imágenes en '{TEST_IMAGES_DIR}'.")
        print("        Agrega imágenes de prueba (.jpg/.png/etc.) y reintenta.")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if APPLY_CLAHE else None

    # Precargar referencias para AMBOS detectores
    cache = {}
    for kind in ("sift", "orb"):
        print(f"\n[INFO] Inicializando {kind.upper()}...")
        det  = create_detector(kind)
        mat  = create_matcher(kind)
        refs = load_references(det, REFERENCES_DIR, clahe, kind_label=kind)
        cache[kind] = (det, mat, refs)

    if not cache["sift"][2] and not cache["orb"][2]:
        print("[ERROR] Ninguna referencia se pudo cargar. Aborta.")
        return

    # Tabla resumen global
    summary = {
        "sift": {"total_kp": 0, "total_time": 0.0,
                 "detected": {k: 0 for k in SIGNS}},
        "orb":  {"total_kp": 0, "total_time": 0.0,
                 "detected": {k: 0 for k in SIGNS}},
    }

    print(f"\n[INFO] Procesando {len(test_files)} imágenes de prueba...\n")

    for fname in test_files:
        path = os.path.join(TEST_IMAGES_DIR, fname)
        bgr  = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [AVISO] No se pudo leer '{fname}'. Saltando.")
            continue

        bgr  = resize_max(bgr, MAX_TEST_SIZE)
        base = os.path.splitext(fname)[0]
        print(f"  · {fname}  ({bgr.shape[1]}x{bgr.shape[0]})")

        annotated = {}
        for kind in ("sift", "orb"):
            det, mat, refs = cache[kind]
            if not refs:
                annotated[kind] = bgr.copy()
                draw_header(annotated[kind], f"{kind.upper()} (sin referencias)")
                continue

            img_out, stats = process_image(bgr, det, mat, refs, clahe)

            # Actualizar resumen
            summary[kind]["total_kp"]   += stats["n_keypoints"]
            summary[kind]["total_time"] += stats["t_extract"] + stats["t_match"]
            for lbl, info in stats["per_sign"].items():
                if info["detected"]:
                    summary[kind]["detected"][lbl] += 1

            # Header con info del detector
            header = (f"{kind.upper()}  |  {stats['n_keypoints']} kpts  |  "
                      f"{(stats['t_extract']+stats['t_match'])*1000:.0f} ms")
            draw_header(img_out, header)
            annotated[kind] = img_out

            # Guardar salida individual
            out_path = os.path.join(OUTPUTS_DIR, f"{base}_{kind}.png")
            cv2.imwrite(out_path, img_out)

            # Imprimir detalles por señal
            dets = [(lbl, info) for lbl, info in stats["per_sign"].items()
                    if info["detected"]]
            if dets:
                for lbl, info in dets:
                    print(f"      [{kind.upper():4s}]  ✓ {lbl:17s}  "
                          f"good={info['n_good']:3d}  inliers={info['n_inliers']:3d}")
            else:
                # Mostrar el candidato con más matches aunque no haya pasado
                best = max(stats["per_sign"].items(),
                           key=lambda kv: kv[1]["n_good"])
                print(f"      [{kind.upper():4s}]  (sin detección — mejor candidato: "
                      f"{best[0]}, good={best[1]['n_good']})")

        # Comparativa lado a lado SIFT | ORB
        s_img = annotated["sift"]
        o_img = annotated["orb"]
        # Igualar alturas si por algún motivo difieren
        if s_img.shape[0] != o_img.shape[0]:
            h_min = min(s_img.shape[0], o_img.shape[0])
            s_img = cv2.resize(s_img, (int(s_img.shape[1]*h_min/s_img.shape[0]), h_min))
            o_img = cv2.resize(o_img, (int(o_img.shape[1]*h_min/o_img.shape[0]), h_min))
        sep = np.full((s_img.shape[0], 4, 3), 255, dtype=np.uint8)
        comp = np.hstack([s_img, sep, o_img])
        cv2.imwrite(os.path.join(OUTPUTS_DIR, f"{base}_comparison.png"), comp)

        print()

    # ======================================
    # RESUMEN GLOBAL
    # ======================================
    n = len(test_files)
    print("=" * 68)
    print("RESUMEN GLOBAL")
    print("=" * 68)
    for kind in ("sift", "orb"):
        s = summary[kind]
        avg_kp   = s["total_kp"]   / n if n else 0
        avg_time = s["total_time"] / n if n else 0
        print(f"\n  {kind.upper()}")
        print(f"    Tiempo total: {s['total_time']:.2f} s   "
              f"(promedio {avg_time*1000:.0f} ms/imagen)")
        print(f"    Keypoints promedio: {avg_kp:.0f}")
        print(f"    Detecciones por señal (de {n} imágenes):")
        for lbl in SIGNS:
            c = s["detected"][lbl]
            pct = c / n * 100 if n else 0
            print(f"      {lbl:18s}: {c}/{n}  ({pct:.0f}%)")

    print("\n[INFO] Resultados guardados en:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
