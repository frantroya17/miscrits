import time
import threading
import re
import os
import difflib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
from mss import mss
import pytesseract
try:
    import easyocr
except Exception:
    easyocr = None
import keyboard


MONITOR = 1
SLEEP_SEC = 0.25
THRESH_DEFAULT = 0.86
CLICK_DELAY = 0.15
ACTION_COOLDOWN_SEC = 1.0
OCR_COOLDOWN_SEC = 0.5
NAME_OCR_COOLDOWN_SEC = 0.75
RARITY_TPL_THRESHOLD = 0.86
WORLD_CLICK_TPL_THRESHOLD = 0.88
WINDOW_TITLE = "Miscrits"
WORLD_DOUBLE_CLICK_DELAY = 0.12
CLICK_HOLD_SEC = 0.05
CLICK_USE_PRESS = True
RARITY_PHASH_THRESHOLD = 10
RARITY_PHASH_ROI = ((735, 50), (762, 74))
RARITY_COLOR_ROI = ((738, 52), (741, 55))
RARITY_COLOR_MAX_DIST = 120.0
RARITY_COLOR_SAT_MAX_COMMON = 35.0
RARITY_COLOR_SAT_MIN_COLOR = 50.0
RARITY_COLOR_HUE_MAX_DIST = 22.0
TRAINING_TPL_THRESHOLD = 0.98
TRAINING_BTN_THRESHOLD = 0.92
BTN_BIEN_THRESHOLD = 0.9
LOG_FILE = "bot_logs.txt"
LOG_MAX_LINES = 2000
MISCRITS_CATALOG_FILE = "miscrits.txt"
MISCRITS_CAPTURABLE_FILE = "miscrits_capturables.txt"
OCR_USER_WORDS_FILE = "tesseract_user_words.txt"
OCR_NAME_SCALE = 4
OCR_NAME_FUZZY_CUTOFF = 0.72
EASYOCR_LANGS = ["en"]
EASYOCR_GPU = False

UI_BG = "#0b0d10"
UI_FG = "#f6f1e9"
UI_MUTED = "#b39b7f"
UI_CARD = "#15121a"
UI_ACCENT = "#ff5c7a"

pyautogui.FAILSAFE = True

RARITIES = [
    ("Común", "comun"),
    ("Raro", "raro"),
    ("Épico", "epico"),
    ("Exótico", "exotico"),
    ("Legendario", "legendario"),
]


def configure_tesseract() -> Optional[str]:
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return cmd
    return None


BOT_COORDS = {
    "tech_left_arrow": (141,697),
    "tech_right_arrow": (1037,698),
    "tech_slots": [
        (280, 700),
        (497, 700),
        (707, 700),
        (899, 700),
    ],
    "capture_ocr_rect": [(561,162), (585,178)],
    "miscrit_name_rect": [(830,50), (934,74)],
    "neutral_hover_pos": (30, 30),
}


def get_game_window():
    wins = gw.getWindowsWithTitle(WINDOW_TITLE)
    exact = [w for w in wins if w.title == WINDOW_TITLE]
    if exact:
        wins = exact
    if not wins:
        return None
    w = wins[0]
    if w.isMinimized:
        w.restore()
        time.sleep(0.2)
    return w


def focus_game_window() -> bool:
    w = get_game_window()
    if not w:
        return False
    try:
        if not w.isActive:
            w.activate()
            time.sleep(0.1)
    except Exception:
        return False
    return True


def update_click_settings(use_press: bool, hold_ms: int) -> None:
    global CLICK_USE_PRESS, CLICK_HOLD_SEC
    CLICK_USE_PRESS = bool(use_press)
    CLICK_HOLD_SEC = max(0.0, min(1.0, hold_ms / 1000))


def click_rel(rel_x, rel_y) -> bool:
    w = get_game_window()
    if not w:
        return False
    if not focus_game_window():
        return False
    abs_x = w.left + rel_x
    abs_y = w.top + rel_y
    pyautogui.moveTo(abs_x, abs_y)
    if CLICK_USE_PRESS:
        pyautogui.mouseDown()
        time.sleep(CLICK_HOLD_SEC)
        pyautogui.mouseUp()
    else:
        pyautogui.click()
    time.sleep(CLICK_DELAY)
    return True


def move_rel(rel_x, rel_y) -> bool:
    w = get_game_window()
    if not w:
        return False
    if not focus_game_window():
        return False
    abs_x = w.left + rel_x
    abs_y = w.top + rel_y
    pyautogui.moveTo(abs_x, abs_y)
    return True


@dataclass
class Template:
    path: str
    threshold: float = THRESH_DEFAULT


def load_tpl(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar template: {path}")
    return img


def match(screen_gray: np.ndarray, tpl_gray: np.ndarray) -> float:
    res = cv2.matchTemplate(screen_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
    return float(max_val)


def find_center(screen_gray: np.ndarray, tpl_gray: np.ndarray, threshold: float):
    res = cv2.matchTemplate(screen_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        return None
    h, w = tpl_gray.shape[:2]
    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2
    return (cx, cy, float(max_val))


def extract_roi_gray(
    screen_gray: np.ndarray,
    window,
    monitor,
    roi: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Optional[np.ndarray]:
    (x1, y1), (x2, y2) = roi
    left = window.left + min(x1, x2) - monitor["left"]
    right = window.left + max(x1, x2) - monitor["left"]
    top = window.top + min(y1, y2) - monitor["top"]
    bottom = window.top + max(y1, y2) - monitor["top"]

    left = max(0, left)
    top = max(0, top)
    right = min(screen_gray.shape[1], right)
    bottom = min(screen_gray.shape[0], bottom)

    if right <= left or bottom <= top:
        return None
    crop = screen_gray[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def extract_roi_bgr(
    screen_bgr: np.ndarray,
    window,
    monitor,
    roi: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Optional[np.ndarray]:
    (x1, y1), (x2, y2) = roi
    left = window.left + min(x1, x2) - monitor["left"]
    right = window.left + max(x1, x2) - monitor["left"]
    top = window.top + min(y1, y2) - monitor["top"]
    bottom = window.top + max(y1, y2) - monitor["top"]

    left = max(0, left)
    top = max(0, top)
    right = min(screen_bgr.shape[1], right)
    bottom = min(screen_bgr.shape[0], bottom)

    if right <= left or bottom <= top:
        return None
    crop = screen_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def rect_to_abs(window, monitor, rect: Tuple[Tuple[int, int], Tuple[int, int]]):
    (x1, y1), (x2, y2) = rect
    left = window.left + min(x1, x2) - monitor["left"]
    right = window.left + max(x1, x2) - monitor["left"]
    top = window.top + min(y1, y2) - monitor["top"]
    bottom = window.top + max(y1, y2) - monitor["top"]
    return int(left), int(top), int(right), int(bottom)


def mean_bgr(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr.reshape(-1, 3).mean(axis=0)


def mean_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


def circular_mean_hue(hues: np.ndarray) -> float:
    if hues.size == 0:
        return 0.0
    angles = hues.astype(np.float32) * (2.0 * np.pi / 180.0)
    sin_mean = np.sin(angles).mean()
    cos_mean = np.cos(angles).mean()
    angle = np.arctan2(sin_mean, cos_mean)
    if angle < 0:
        angle += 2.0 * np.pi
    hue = (angle * 180.0 / np.pi) / 2.0
    return float(hue)


def hue_distance(a: float, b: float) -> float:
    dh = abs(a - b)
    return float(min(dh, 180.0 - dh))


def phash(img_gray: np.ndarray, size: int = 32, hash_size: int = 8) -> np.ndarray:
    resized = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def normalize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z ]+", " ", name)
    clean = re.sub(r"\s+", " ", clean).strip().lower()
    return clean


def load_names_from_file(path: str) -> list:
    if not os.path.exists(path):
        return []
    names = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                names.append(name)
    except OSError:
        return []
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name in names:
        key = normalize_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def save_names_to_file(path: str, names: list) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for name in names:
                handle.write(name.strip() + "\n")
    except OSError:
        pass


def build_user_words_file(path: str, names: list) -> Optional[str]:
    if not names:
        return None
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for name in names:
                clean = name.strip()
                if clean:
                    handle.write(clean + "\n")
        return path
    except OSError:
        return None


def preprocess_name_ocr(crop: np.ndarray) -> np.ndarray:
    scaled = cv2.resize(
        crop,
        None,
        fx=OCR_NAME_SCALE,
        fy=OCR_NAME_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(scaled)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3,
    )
    return thresh


def preprocess_name_variants(crop: np.ndarray):
    scaled = cv2.resize(
        crop,
        None,
        fx=OCR_NAME_SCALE,
        fy=OCR_NAME_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(scaled)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    adapt = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3,
    )
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        ("adapt", adapt),
        ("adapt_inv", cv2.bitwise_not(adapt)),
        ("otsu", otsu),
        ("otsu_inv", cv2.bitwise_not(otsu)),
    ]


def click_at(x, y):
    if not focus_game_window():
        return
    pyautogui.moveTo(x, y)
    if CLICK_USE_PRESS:
        pyautogui.mouseDown()
        time.sleep(CLICK_HOLD_SEC)
        pyautogui.mouseUp()
    else:
        pyautogui.click()
    time.sleep(CLICK_DELAY)


def detect_state(screen_gray: np.ndarray, tpls: Dict[str, Tuple[np.ndarray, float]]) -> Optional[str]:
    for state in ["CAPTURED_POPUP", "VICTORY"]:
        tpl_img, thr = tpls[state]
        if match(screen_gray, tpl_img) >= thr:
            return state

    for state in ["FIGHT_MY_TURN", "FIGHT_WAIT"]:
        tpl_img, thr = tpls[state]
        if match(screen_gray, tpl_img) >= thr:
            return state

    tpl_img, thr = tpls["WORLD"]
    if match(screen_gray, tpl_img) >= thr:
        return "WORLD"

    return None


class BotRunner:
    def __init__(self, on_state=None, on_log=None, on_capture_rate=None):
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.on_state = on_state
        self.on_log = on_log
        self.on_capture_rate = on_capture_rate
        self.easyocr_reader = None
        self.tesseract_cmd = None
        if easyocr is not None:
            try:
                self.easyocr_reader = easyocr.Reader(
                    EASYOCR_LANGS, gpu=EASYOCR_GPU, verbose=False
                )
                self.log(
                    f"[OCR] Usando EasyOCR (langs={EASYOCR_LANGS}, gpu={EASYOCR_GPU})"
                )
            except Exception as exc:
                self.easyocr_reader = None
                self.log(f"[OCR] EasyOCR no disponible: {exc}")
        if self.easyocr_reader is None:
            self.tesseract_cmd = configure_tesseract()
            if self.tesseract_cmd:
                self.log(f"[OCR] Usando Tesseract: {self.tesseract_cmd}")
            else:
                self.log(f"[OCR] Tesseract cmd: {pytesseract.pytesseract.tesseract_cmd}")

        self.last_action_time = 0.0
        self.last_ocr_time = 0.0
        self.last_state = None
        self.last_capture_rate = None
        self.last_world_click_time = 0.0
        self.last_miscrit_name = None
        self.last_name_time = 0.0

        # Config (desde UI)
        self.auto_continue = True
        self.auto_save = True
        self.kill_attack_index = 1      # 1..12
        self.capture_attack_index = 1   # 1..12  <-- NUEVO
        self.capture_success_rate = 50  # 1..100 (%)
        self.world_click_cooldown_sec = 30
        self.world_click_template_path: Optional[str] = None
        self.world_click_template: Optional[np.ndarray] = None
        self.world_click_double = False
        self.last_rarity = None
        self.name_capturable = set()
        self.click_use_press = True
        self.click_hold_ms = 50
        self.train_plat_enabled = False
        self.miscrit_catalog = load_names_from_file(MISCRITS_CATALOG_FILE)
        self.miscrit_catalog_norm = [
            normalize_name(n) for n in self.miscrit_catalog if normalize_name(n)
        ]
        self.user_words_path = build_user_words_file(
            OCR_USER_WORDS_FILE, self.miscrit_catalog
        )

        # Coordenadas (RELATIVAS)
        self.tech_left_arrow = BOT_COORDS["tech_left_arrow"]
        self.tech_right_arrow = BOT_COORDS["tech_right_arrow"]
        self.tech_slots = BOT_COORDS["tech_slots"]
        self.capture_ocr_rect = BOT_COORDS["capture_ocr_rect"]
        self.miscrit_name_rect = BOT_COORDS["miscrit_name_rect"]
        self.neutral_hover_pos = BOT_COORDS["neutral_hover_pos"]

        # Templates de estado
        self.templates = {
            "WORLD": Template("tpl/world.png", 0.86),
            "CAPTURED_POPUP": Template("tpl/captured.png", 0.86),
            "FIGHT_MY_TURN": Template("tpl/my_turn.png", 0.86),
            "FIGHT_WAIT": Template("tpl/wait.png", 0.86),
            "VICTORY": Template("tpl/victory.png", 0.86),
        }

        self.tpls_loaded: Dict[str, Tuple[np.ndarray, float]] = {}
        for k, t in self.templates.items():
            self.tpls_loaded[k] = (load_tpl(t.path), t.threshold)

        # Botones popup (por template)
        self.btn_continue = load_tpl("tpl/btn_continue.png")
        self.btn_continue_alt = load_tpl("tpl/continue.png")
        self.btn_save = load_tpl("tpl/btn_save.png")
        self.btn_capture = load_tpl("tpl/captura.png")
        self.btn_training = load_tpl("tpl/entrenamiento.png")
        self.btn_training_ready = load_tpl("tpl/puede_entrenar.png")
        self.btn_train = load_tpl("tpl/entrenar.png")
        self.btn_training_continue = load_tpl("tpl/continuar_entrenamiento.png")
        self.btn_training_exit = load_tpl("tpl/salir_entrenar.png")
        self.btn_train_plat = load_tpl("tpl/add_bonus_entrenamiento.png")
        self.btn_bien = load_tpl("tpl/btn_bien.png")
        self.btn_thr = 0.85
        self.btn_capture_thr = 0.85

        # Templates de rareza


    def _fuzzy_fix_name(self, norm_name: str) -> Tuple[str, float]:
        if not norm_name or not self.miscrit_catalog_norm:
            return norm_name, 0.0
        if norm_name in self.miscrit_catalog_norm:
            return norm_name, 1.0
        best = None
        best_score = 0.0
        for cand in self.miscrit_catalog_norm:
            score = difflib.SequenceMatcher(None, norm_name, cand).ratio()
            if score > best_score:
                best = cand
                best_score = score
        if best and best_score >= OCR_NAME_FUZZY_CUTOFF:
            return best, best_score
        return norm_name, best_score

    def log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def set_state(self, st: str):
        if self.on_state:
            self.on_state(st)

    def clear_hover(self):
        moved = move_rel(*self.neutral_hover_pos)
        if not moved:
            self.log("[WARN] No encuentro la ventana 'Miscrits' para quitar hover.")

    def set_capture_rate(self, rate: Optional[int]):
        if rate != self.last_capture_rate:
            self.last_capture_rate = rate
            if self.on_capture_rate:
                self.on_capture_rate(rate)
            if rate is None:
                self.log("[OCR] Captura: --%")
            else:
                self.log(f"[OCR] Captura: {rate}%")

    def _detect_rarity(
        self,
        screen_gray: np.ndarray,
        screen_bgr: Optional[np.ndarray] = None,
        window=None,
        monitor=None,
    ) -> Optional[str]:
        if self.rarity_colors:
            if screen_bgr is not None and window and monitor:
                roi = extract_roi_bgr(screen_bgr, window, monitor, RARITY_COLOR_ROI)
                if roi is not None:
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    sat_median = float(np.median(roi_hsv[:, :, 1]))
                    if (
                        sat_median <= RARITY_COLOR_SAT_MAX_COMMON
                        and "comun" in self.rarity_colors
                    ):
                        if self.last_rarity != "comun":
                            self.log("[RARITY] Detectada: comun (sat low)")
                            self.last_rarity = "comun"
                        return "comun"
                    sat_mask = roi_hsv[:, :, 1] >= RARITY_COLOR_SAT_MIN_COLOR
                    if np.any(sat_mask):
                        hues = roi_hsv[:, :, 0][sat_mask]
                    else:
                        hues = roi_hsv[:, :, 0].reshape(-1)
                    roi_hue = circular_mean_hue(hues)
                    best_key = None
                    best_dist = 10**9
                    for key, ref_color in self.rarity_colors.items():
                        if key == "comun":
                            continue
                        dist = hue_distance(roi_hue, float(ref_color[0]))
                        if dist < best_dist:
                            best_dist = dist
                            best_key = key
                    if best_key is not None and best_dist <= RARITY_COLOR_HUE_MAX_DIST:
                        if best_key != self.last_rarity:
                            self.log(f"[RARITY] Detectada: {best_key} (h={best_dist:.1f})")
                            self.last_rarity = best_key
                        return best_key
                    if self.last_rarity != "unknown":
                        self.log(f"[RARITY] Unknown (h={best_dist:.1f})")
                        self.last_rarity = "unknown"
                    return None
                if self.last_rarity != "unknown":
                    self.log("[RARITY] ROI invalida para color")
                    self.last_rarity = "unknown"
            else:
                if self.last_rarity != "unknown":
                    self.log("[RARITY] Color no disponible (sin ventana/buffer)")
                    self.last_rarity = "unknown"
            return None

        if self.rarity_hashes and window and monitor:
            roi = extract_roi_gray(screen_gray, window, monitor, RARITY_PHASH_ROI)
            if roi is not None:
                roi_hash = phash(roi)
                best_key = None
                best_dist = 10**9
                for key, ref_hash in self.rarity_hashes.items():
                    dist = hamming_distance(roi_hash, ref_hash)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = key
                if best_key is not None and best_dist <= RARITY_PHASH_THRESHOLD:
                    if best_key != self.last_rarity:
                        self.log(f"[RARITY] Detectada: {best_key} (d={best_dist})")
                        self.last_rarity = best_key
                    return best_key
                if self.last_rarity != "unknown":
                    self.log(f"[RARITY] Unknown (d={best_dist})")
                    self.last_rarity = "unknown"
                return None
            if self.last_rarity != "unknown":
                self.log("[RARITY] ROI invalida para pHash")
                self.last_rarity = "unknown"
            return None

        best_key = None
        best_score = 0.0
        for key, (tpl_img, thr) in self.rarity_tpls.items():
            score = match(screen_gray, tpl_img)
            if score >= thr and score > best_score:
                best_key = key
                best_score = score
        if best_key and best_key != self.last_rarity:
            self.log(f"[RARITY] Detectada: {best_key} ({best_score:.2f})")
            self.last_rarity = best_key
        return best_key

    def _easyocr_best_text(self, img: np.ndarray, allowlist: Optional[str] = None):
        if self.easyocr_reader is None:
            return None
        try:
            results = self.easyocr_reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
            )
        except Exception as exc:
            self.log(f"[OCR] Error EasyOCR: {exc}")
            return None
        if not results:
            return None
        best = max(results, key=lambda r: r[2] if len(r) > 2 else 0.0)
        text = (best[1] or "").strip()
        conf = float(best[2]) if len(best) > 2 else 0.0
        if not text:
            return None
        return text, conf

    def _read_capture_rate(self, screen_gray: np.ndarray, window, monitor) -> Optional[int]:
        rect = self.capture_ocr_rect
        left = window.left + min(rect[0][0], rect[1][0]) - monitor["left"]
        right = window.left + max(rect[0][0], rect[1][0]) - monitor["left"]
        top = window.top + min(rect[0][1], rect[1][1]) - monitor["top"]
        bottom = window.top + max(rect[0][1], rect[1][1]) - monitor["top"]

        left = max(0, left)
        top = max(0, top)
        right = min(screen_gray.shape[1], right)
        bottom = min(screen_gray.shape[0], bottom)

        if right <= left or bottom <= top:
            return None

        crop = screen_gray[top:bottom, left:right]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.easyocr_reader is not None:
            result = self._easyocr_best_text(thresh, allowlist="0123456789")
            if not result:
                self.log("[OCR] EasyOCR: '' -> --%")
                return None
            raw_text, conf = result
            digits = re.findall(r"\d+", raw_text)
            if not digits:
                self.log(f"[OCR] EasyOCR: '{raw_text}' ({conf:.2f}) -> --%")
                return None
            value = int(digits[0])
            value = max(0, min(100, value))
            self.log(f"[OCR] EasyOCR: '{raw_text}' ({conf:.2f}) -> {value}%")
            return value
        try:
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
            )
        except Exception as exc:
            self.log(f"[OCR] Error Tesseract: {exc}")
            return None
        raw_text = text.strip()
        digits = re.findall(r"\d+", text)
        if not digits:
            self.log(f"[OCR] Texto: '{raw_text}' -> --%")
            return None
        value = int(digits[0])
        value = max(0, min(100, value))
        self.log(f"[OCR] Texto: '{raw_text}' -> {value}%")
        return value

    def save_ocr_debug_image(self) -> Optional[str]:
        window = get_game_window()
        if not window:
            self.log("[OCR] No encuentro la ventana 'Miscrits' para debug OCR.")
            return None
        self.log(
            "[OCR] Ventana Miscrits "
            f"(left={window.left}, top={window.top}, "
            f"width={window.width}, height={window.height})"
        )
        try:
            with mss() as sct:
                monitor = sct.monitors[MONITOR]
                img = np.array(sct.grab(monitor))
        except Exception as exc:
            self.log(f"[OCR] Error capturando pantalla: {exc}")
            return None

        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]

        rects = [
            ("capture_ocr", self.capture_ocr_rect, (0, 255, 0)),
            ("miscrit_name", self.miscrit_name_rect, (0, 0, 255)),
        ]
        for label, rect, color in rects:
            left, top, right, bottom = rect_to_abs(window, monitor, rect)
            left = max(0, min(w - 1, left))
            right = max(0, min(w - 1, right))
            top = max(0, min(h - 1, top))
            bottom = max(0, min(h - 1, bottom))
            if right <= left or bottom <= top:
                continue
            cv2.rectangle(bgr, (left, top), (right, bottom), color, 2)
            cv2.putText(
                bgr,
                label,
                (left, max(0, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        filename = f"ocr_debug_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(os.getcwd(), filename)
        if cv2.imwrite(path, bgr):
            self.log(f"[OCR] Debug guardado: {path}")
            return path
        self.log("[OCR] No se pudo guardar debug OCR.")
        return None

    def _read_miscrit_name(self, screen_gray: np.ndarray, window, monitor) -> Optional[str]:
        rect = self.miscrit_name_rect
        left = window.left + min(rect[0][0], rect[1][0]) - monitor["left"]
        right = window.left + max(rect[0][0], rect[1][0]) - monitor["left"]
        top = window.top + min(rect[0][1], rect[1][1]) - monitor["top"]
        bottom = window.top + max(rect[0][1], rect[1][1]) - monitor["top"]

        left = max(0, left)
        top = max(0, top)
        right = min(screen_gray.shape[1], right)
        bottom = min(screen_gray.shape[0], bottom)

        if right <= left or bottom <= top:
            return None

        crop = screen_gray[top:bottom, left:right]
        if crop.size == 0:
            return None

        best_norm = ""
        best_raw = ""
        best_score = 0.0
        best_label = ""
        if self.easyocr_reader is not None:
            best_conf = 0.0
            for label, img in preprocess_name_variants(crop):
                result = self._easyocr_best_text(
                    img,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                )
                if not result:
                    continue
                raw_text, conf = result
                norm = normalize_name(raw_text)
                if not norm:
                    continue
                fixed, score = self._fuzzy_fix_name(norm)
                if score > best_score or (score == best_score and conf > best_conf):
                    best_norm = fixed
                    best_raw = raw_text
                    best_score = score
                    best_label = f"{label}/easyocr"
                    best_conf = conf
        else:
            config = (
                "--oem 1 --psm 8 -l eng "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
            )
            if self.user_words_path:
                config += f' --user-words "{self.user_words_path}"'
            for label, img in preprocess_name_variants(crop):
                try:
                    text = pytesseract.image_to_string(img, config=config)
                except Exception as exc:
                    self.log(f"[OCR] Error Tesseract (nombre): {exc}")
                    continue
                raw_text = text.strip()
                norm = normalize_name(raw_text)
                if not norm:
                    continue
                fixed, score = self._fuzzy_fix_name(norm)
                if score > best_score:
                    best_norm = fixed
                    best_raw = raw_text
                    best_score = score
                    best_label = label

        if not best_norm:
            self.log("[OCR] Nombre: '' -> --")
            return None
        self.log(
            f"[OCR] Nombre({best_label}): '{best_raw}' -> {best_norm} ({best_score:.2f})"
        )
        return best_norm

    def start(self):
        if self.running:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.running = True
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.running = False

    def _select_attack_by_index(self, attack_index: int):
        """Selección genérica 1..12 por flechas y 4 slots (REL)."""
        idx = max(1, min(12, int(attack_index)))
        page = (idx - 1) // 4
        slot = (idx - 1) % 4

        if not get_game_window():
            self.log("[WARN] No encuentro la ventana 'Miscrits'.")
            return

        for _ in range(3):
            click_rel(*self.tech_left_arrow)
            time.sleep(0.10)

        for _ in range(page):
            click_rel(*self.tech_right_arrow)
            time.sleep(0.12)

        click_rel(*self.tech_slots[slot])

    def select_kill_attack(self, attack_index: int):
        self._select_attack_by_index(attack_index)

    def select_capture_attack(self, attack_index: int):
        # Por ahora EXACTAMENTE igual que matar
        self._select_attack_by_index(attack_index)

    def set_world_click_template(self, path: Optional[str]):
        if not path:
            self.world_click_template_path = None
            self.world_click_template = None
            return
        self.world_click_template_path = path
        self.world_click_template = load_tpl(path)

    def apply_click_settings(self):
        update_click_settings(self.click_use_press, self.click_hold_ms)

    def _run_loop(self):
        self.log("[BOT] Iniciado")
        try:
            sct = mss()
            monitor = sct.monitors[MONITOR]
            fight_states = {"FIGHT_WAIT", "FIGHT_MY_TURN"}

            while not self.stop_event.is_set():
                img = np.array(sct.grab(monitor))
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

                state = detect_state(gray, self.tpls_loaded)
                if state is None:
                    state = self.last_state

                if state != self.last_state and state is not None:
                    self.log(f"[STATE] {state}")
                    self.set_state(state)
                    if state == "FIGHT_WAIT":
                        self.clear_hover()
                    if state in fight_states and self.last_state not in fight_states:
                        self.last_rarity = None
                    self.last_state = state

                now = time.time()
                can_act = (now - self.last_action_time) >= ACTION_COOLDOWN_SEC

                if can_act:
                    pos = find_center(gray, self.btn_bien, BTN_BIEN_THRESHOLD)
                    if pos:
                        x, y, conf = pos
                        self.log(f"[ACTION] Click BIEN ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()
                        continue
                    if self.auto_continue:
                        pos = find_center(gray, self.btn_continue, self.btn_thr)
                        if not pos:
                            pos = find_center(gray, self.btn_continue_alt, self.btn_thr)
                        if pos:
                            x, y, conf = pos
                            self.log(f"[ACTION] Click CONTINUAR ({conf:.2f})")
                            click_at(x, y)
                            self.last_action_time = time.time()
                            continue
                    pos = find_center(gray, self.btn_training_ready, TRAINING_BTN_THRESHOLD)
                    if pos:
                        x, y, conf = pos
                        self.log(f"[ACTION] Click PUEDE ENTRENAR ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()
                        continue
                    pos = find_center(gray, self.btn_train, TRAINING_BTN_THRESHOLD)
                    if pos:
                        x, y, conf = pos
                        self.log(f"[ACTION] Click ENTRENAR ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()
                        continue
                    training_continue_tpl = (
                        self.btn_train_plat if self.train_plat_enabled else self.btn_training_continue
                    )
                    pos = find_center(gray, training_continue_tpl, TRAINING_BTN_THRESHOLD)
                    if pos:
                        x, y, conf = pos
                        action_label = (
                            "CLICK ENTRENAR PLATINO"
                            if self.train_plat_enabled
                            else "CLICK CONTINUAR ENTRENAMIENTO"
                        )
                        self.log(f"[ACTION] {action_label} ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()
                        continue
                    pos = find_center(gray, self.btn_training_exit, TRAINING_BTN_THRESHOLD)
                    if pos:
                        x, y, conf = pos
                        self.log(f"[ACTION] Click SALIR ENTRENAR ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()
                        continue

                if can_act and state == "VICTORY" and self.auto_continue:
                    pos = find_center(gray, self.btn_continue, self.btn_thr)
                    if not pos:
                        pos = find_center(gray, self.btn_continue_alt, self.btn_thr)
                    if pos:
                        x, y, conf = pos
                        self.log(f"[ACTION] Click CONTINUAR ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()

                elif can_act and state == "CAPTURED_POPUP" and self.auto_save:
                    pos = find_center(gray, self.btn_save, self.btn_thr)
                    if pos:
                        x, y, conf = pos
                        self.log(f"[ACTION] Click GUARDAR ({conf:.2f})")
                        click_at(x, y)
                        self.last_action_time = time.time()

                elif can_act and state == "FIGHT_MY_TURN":
                    window = get_game_window()
                    use_name_filter = bool(self.name_capturable)
                    is_capturable = False

                    if use_name_filter:
                        name = None
                        if (time.time() - self.last_name_time) >= NAME_OCR_COOLDOWN_SEC:
                            if window:
                                name = self._read_miscrit_name(gray, window, monitor)
                                self.last_name_time = time.time()
                                self.last_miscrit_name = name
                            else:
                                self.last_miscrit_name = None
                        else:
                            name = self.last_miscrit_name

                        is_capturable = bool(name and name in self.name_capturable)
                        if not name:
                            self.log("[OCR] Nombre no detectado, trato como no capturable")
                    else:
                        self.log("[OCR] Lista de capturables vacia, trato como no capturable")

                    if is_capturable:
                        rate = None
                        if (time.time() - self.last_ocr_time) >= OCR_COOLDOWN_SEC:
                            if window:
                                rate = self._read_capture_rate(gray, window, monitor)
                                self.set_capture_rate(rate)
                                self.last_ocr_time = time.time()
                            else:
                                self.set_capture_rate(None)
                        else:
                            rate = self.last_capture_rate

                        if rate is not None and rate >= self.capture_success_rate:
                            pos = find_center(gray, self.btn_capture, self.btn_capture_thr)
                            if pos:
                                x, y, conf = pos
                                self.log(
                                    f"[ACTION] Capturar (>= {self.capture_success_rate}%) "
                                    f"({conf:.2f})"
                                )
                                click_at(x, y)
                                self.last_action_time = time.time()
                                continue
                            self.log(
                                f"[WARN] No encuentro boton CAPTURA, uso ataque "
                                f"#{self.capture_attack_index}"
                            )
                        else:
                            self.log(
                                f"[ACTION] Ataque CAPTURABLE (< {self.capture_success_rate}%) "
                                f"#{self.capture_attack_index}"
                            )
                        try:
                            img_check = np.array(sct.grab(monitor))
                            gray_check = cv2.cvtColor(img_check, cv2.COLOR_BGRA2GRAY)
                            state_check = detect_state(gray_check, self.tpls_loaded)
                            if state_check != "FIGHT_MY_TURN":
                                self.log("[WARN] Estado cambiado, no selecciono ataque")
                                self.last_action_time = time.time()
                                continue
                        except Exception as exc:
                            self.log(f"[WARN] No pude revalidar estado: {exc}")
                        self.select_capture_attack(self.capture_attack_index)
                    else:
                        self.log(f"[ACTION] Ataque MATAR #{self.kill_attack_index}")
                        try:
                            img_check = np.array(sct.grab(monitor))
                            gray_check = cv2.cvtColor(img_check, cv2.COLOR_BGRA2GRAY)
                            state_check = detect_state(gray_check, self.tpls_loaded)
                            if state_check != "FIGHT_MY_TURN":
                                self.log("[WARN] Estado cambiado, no selecciono ataque")
                                self.last_action_time = time.time()
                                continue
                        except Exception as exc:
                            self.log(f"[WARN] No pude revalidar estado: {exc}")
                        self.select_kill_attack(self.kill_attack_index)

                    self.last_action_time = time.time()

                if state == "WORLD":
                    if can_act:
                        pos = find_center(gray, self.btn_train, 0.86)
                        if pos:
                            x, y, conf = pos
                            self.log(f"[ACTION] Click ENTRENAR ({conf:.2f})")
                            click_at(x, y)
                            self.last_action_time = time.time()
                            continue
                        pos = find_center(gray, self.btn_training_ready, TRAINING_BTN_THRESHOLD)
                        if pos:
                            x, y, conf = pos
                            self.log(f"[ACTION] Click PUEDE ENTRENAR ({conf:.2f})")
                            click_at(x, y)
                            self.last_action_time = time.time()
                            continue
                        pos = find_center(gray, self.btn_training, TRAINING_TPL_THRESHOLD)
                        if pos:
                            x, y, conf = pos
                            self.log(f"[ACTION] Click ENTRENAMIENTO ({conf:.2f})")
                            click_at(x, y)
                            self.last_action_time = time.time()
                            time.sleep(SLEEP_SEC)
                            continue
                        training_continue_tpl = (
                            self.btn_train_plat if self.train_plat_enabled else self.btn_training_continue
                        )
                        pos = find_center(gray, training_continue_tpl, TRAINING_BTN_THRESHOLD)
                        if pos:
                            x, y, conf = pos
                            action_label = (
                                "CLICK ENTRENAR PLATINO"
                                if self.train_plat_enabled
                                else "CLICK CONTINUAR ENTRENAMIENTO"
                            )
                            self.log(f"[ACTION] {action_label} ({conf:.2f})")
                            click_at(x, y)
                            self.last_action_time = time.time()
                            continue
                        pos = find_center(gray, self.btn_training_exit, TRAINING_BTN_THRESHOLD)
                        if pos:
                            x, y, conf = pos
                            self.log(f"[ACTION] Click SALIR ENTRENAR ({conf:.2f})")
                            click_at(x, y)
                            self.last_action_time = time.time()
                            continue
                        
                    if (now - self.last_world_click_time) >= self.world_click_cooldown_sec:
                        if self.world_click_template is None:
                            self.last_world_click_time = now
                        else:
                            pos = find_center(gray, self.world_click_template, WORLD_CLICK_TPL_THRESHOLD)
                            if pos:
                                x, y, conf = pos
                                self.log(
                                    "[ACTION] Click WORLD template "
                                    f"({conf:.2f})"
                                )
                                click_at(x, y)
                                if self.world_click_double:
                                    time.sleep(WORLD_DOUBLE_CLICK_DELAY)
                                    click_at(x, y)
                                self.last_world_click_time = now

                time.sleep(SLEEP_SEC)

        except Exception as e:
            self.log(f"[ERROR] {e}")
        finally:
            self.running = False
            self.log("[BOT] Parado")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Miscrits Bot")
        self.geometry("980x720")
        self.minsize(940, 680)
        self.resizable(True, True)
        self.configure(bg=UI_BG)

        self.log_file_path = os.path.join(os.getcwd(), LOG_FILE)
        self._log_lines = []

        self.state_var = tk.StringVar(value="Estado: -")
        self.running_var = tk.StringVar(value="Parado")
        self.capture_ocr_label_var = tk.StringVar(value="OCR: --%")

        self._configure_style()
        self._reset_log_file()

        top = ttk.Frame(self, padding=10, style="App.TFrame")
        top.pack(fill="x")

        ttk.Label(top, textvariable=self.running_var, style="Title.TLabel").pack(anchor="w")
        ttk.Label(top, textvariable=self.state_var, style="Status.TLabel").pack(anchor="w", pady=(4, 10))

        btns = ttk.Frame(top, style="App.TFrame")
        btns.pack(fill="x")

        self.btn_start = ttk.Button(btns, text="Start", command=self.on_start)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        ttk.Label(top, text="FailSafe: mueve el ratón a la esquina sup-izq para parar").pack(anchor="w", pady=(8, 0))

        opts = ttk.LabelFrame(top, text="Opciones", padding=8, style="App.TLabelframe")
        opts.pack(fill="x", pady=(10, 0))

        combat_frame = ttk.LabelFrame(opts, text="Combate", padding=8, style="App.TLabelframe")
        combat_frame.pack(fill="x", pady=(0, 8))

        combat_row1 = ttk.Frame(combat_frame, style="App.TFrame")
        combat_row1.pack(fill="x", anchor="w")

        ttk.Label(combat_row1, text="Ataque MATAR (1-12):").pack(side="left", padx=(0, 6))
        self.kill_combo = ttk.Combobox(
            combat_row1,
            values=[str(i) for i in range(1, 13)],
            width=3,
            state="readonly",
        )
        self.kill_combo.set("1")
        self.kill_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_settings())
        self.kill_combo.pack(side="left")
        ttk.Label(combat_row1, text="Ataque CAPTURAR (1-12):").pack(side="left", padx=(14, 6))
        self.capture_combo = ttk.Combobox(
            combat_row1,
            values=[str(i) for i in range(1, 13)],
            width=3,
            state="readonly",
        )
        self.capture_combo.set("1")
        self.capture_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_settings())
        self.capture_combo.pack(side="left")

        capture_frame = ttk.LabelFrame(opts, text="Captura", padding=8, style="App.TLabelframe")
        capture_frame.pack(fill="x", pady=(0, 8))

        capture_row2 = ttk.Frame(capture_frame, style="App.TFrame")
        capture_row2.pack(fill="x", anchor="w")

        ttk.Label(capture_row2, text="Exito Captura (% 1-100):").pack(side="left", padx=(0, 6))
        self.capture_rate_var = tk.StringVar(value="50")
        self.capture_rate_spin = ttk.Spinbox(
            capture_row2,
            from_=1,
            to=100,
            width=5,
            textvariable=self.capture_rate_var,
            command=self._apply_settings,
        )
        self.capture_rate_spin.bind("<FocusOut>", lambda e: self._apply_settings())
        self.capture_rate_spin.bind("<Return>", lambda e: self._apply_settings())
        self.capture_rate_spin.pack(side="left")

        ttk.Label(capture_row2, textvariable=self.capture_ocr_label_var).pack(side="left", padx=(10, 0))
        ttk.Button(
            capture_row2,
            text="Ver OCR",
            command=self._preview_ocr_rects,
        ).pack(side="left", padx=(10, 0))

        miscrit_frame = ttk.LabelFrame(opts, text="Miscrits capturables", padding=8, style="App.TLabelframe")
        miscrit_frame.pack(fill="x", pady=(0, 8))

        miscrit_entry_row = ttk.Frame(miscrit_frame, style="App.TFrame")
        miscrit_entry_row.pack(fill="x", anchor="w")

        ttk.Label(miscrit_entry_row, text="Nombre:").pack(side="left", padx=(0, 6))
        self.miscrit_entry = ttk.Entry(miscrit_entry_row, width=24)
        self.miscrit_entry.pack(side="left")
        self.miscrit_entry.bind("<KeyRelease>", self._on_miscrit_entry_change)
        ttk.Button(
            miscrit_entry_row,
            text="Agregar",
            command=self._add_capturable_miscrit,
        ).pack(side="left", padx=(6, 0))

        miscrit_list_row = ttk.Frame(miscrit_frame, style="App.TFrame")
        miscrit_list_row.pack(fill="x", anchor="w", pady=(6, 0))

        self.miscrit_text = tk.Text(
            miscrit_list_row,
            height=4,
            wrap="word",
            bg=UI_CARD,
            fg=UI_FG,
            relief="flat",
        )
        self.miscrit_text.pack(side="left", fill="x", expand=True)
        self.miscrit_text.bind("<<Modified>>", self._on_capturable_text_modified)

        suggest_frame = ttk.Frame(miscrit_list_row, style="App.TFrame")
        suggest_frame.pack(side="left", padx=(10, 0))
        ttk.Label(suggest_frame, text="Sugerencias").pack(anchor="w")
        self.miscrit_suggest_list = tk.Listbox(
            suggest_frame,
            height=4,
            width=20,
            bg=UI_CARD,
            fg=UI_FG,
            relief="flat",
        )
        self.miscrit_suggest_list.pack()
        self.miscrit_suggest_list.bind("<<ListboxSelect>>", self._on_miscrit_suggest_select)

        training_frame = ttk.LabelFrame(opts, text="Entrenamiento", padding=8, style="App.TLabelframe")
        training_frame.pack(fill="x", pady=(0, 8))

        training_row = ttk.Frame(training_frame, style="App.TFrame")
        training_row.pack(fill="x", anchor="w")

        self.train_plat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            training_row,
            text="Entrenar Platino",
            variable=self.train_plat_var,
            command=self._apply_settings,
        ).pack(side="left")

        world_frame = ttk.LabelFrame(opts, text="World Click", padding=8, style="App.TLabelframe")
        world_frame.pack(fill="x")

        world_row = ttk.Frame(world_frame, style="App.TFrame")
        world_row.pack(fill="x", anchor="w")

        ttk.Label(world_row, text="Click WORLD (imagen):").pack(side="left", padx=(0, 6))
        self.world_click_tpl_var = tk.StringVar(value="Sin imagen seleccionada")
        ttk.Label(world_row, textvariable=self.world_click_tpl_var).pack(side="left")
        ttk.Button(
            world_row,
            text="Seleccionar imagen",
            command=self._select_world_click_template,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            world_row,
            text="Quitar",
            command=self._clear_world_click_template,
        ).pack(side="left", padx=(6, 0))

        ttk.Label(world_row, text="Cooldown WORLD (1-30s):").pack(side="left", padx=(14, 6))
        self.world_click_cooldown_var = tk.StringVar(value="30")
        self.world_click_cooldown_spin = ttk.Spinbox(
            world_row,
            from_=1,
            to=30,
            width=4,
            textvariable=self.world_click_cooldown_var,
            command=self._apply_settings,
        )
        self.world_click_cooldown_spin.bind("<FocusOut>", lambda e: self._apply_settings())
        self.world_click_cooldown_spin.bind("<Return>", lambda e: self._apply_settings())
        self.world_click_cooldown_spin.pack(side="left")

        mid = ttk.Frame(self, padding=(10, 0, 10, 10), style="App.TFrame")
        mid.pack(fill="both", expand=True)

        self.log_box = tk.Text(mid, height=14, wrap="word", bg=UI_CARD, fg=UI_FG, relief="flat")
        self.log_box.pack(fill="both", expand=True)

        self.bot = BotRunner(
            on_state=self._ui_set_state,
            on_log=self._ui_log,
            on_capture_rate=self._ui_set_capture_rate,
        )

        self.miscrit_catalog = load_names_from_file(MISCRITS_CATALOG_FILE)
        self.capturable_miscrits = load_names_from_file(MISCRITS_CAPTURABLE_FILE)
        self._refresh_capturable_text()
        self._update_miscrit_suggestions("")

        self.keyboard_hotkeys_enabled = False
        try:
            keyboard.add_hotkey("f10", self.on_start)
            keyboard.add_hotkey("esc", self.on_close)
            self.keyboard_hotkeys_enabled = True
        except Exception as exc:
            self._ui_log(f"[WARN] Hotkeys desactivados: {exc}")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _preview_ocr_rects(self):
        threading.Thread(target=self.bot.save_ocr_debug_image, daemon=True).start()

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=UI_BG)
        style.configure("TLabel", background=UI_BG, foreground=UI_FG)
        style.configure("TCheckbutton", background=UI_BG, foreground=UI_FG)
        style.configure("Title.TLabel", background=UI_BG, foreground=UI_FG, font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", background=UI_BG, foreground=UI_FG)
        style.configure("Info.TLabel", background=UI_BG, foreground=UI_MUTED)
        style.configure("App.TLabelframe", background=UI_BG, foreground=UI_FG)
        style.configure("App.TLabelframe.Label", background=UI_BG, foreground=UI_MUTED, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(10, 6), background=UI_CARD, foreground=UI_FG)
        style.map("TButton", background=[("active", UI_ACCENT)])
        self.option_add("*TCombobox*Listbox*Background", UI_CARD)
        self.option_add("*TCombobox*Listbox*Foreground", UI_FG)

    def _reset_log_file(self):
        try:
            with open(self.log_file_path, "w", encoding="utf-8") as handle:
                handle.write("")
        except OSError:
            pass

    def _write_log_file(self):
        try:
            with open(self.log_file_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self._log_lines))
                if self._log_lines:
                    handle.write("\n")
        except OSError:
            pass

    def _apply_settings(self):
        self.bot.auto_continue = True
        self.bot.auto_save = True
        self.bot.kill_attack_index = int(self.kill_combo.get())
        self.bot.capture_attack_index = int(self.capture_combo.get())
        self.bot.capture_success_rate = self._get_capture_rate()
        self.bot.world_click_cooldown_sec = self._get_world_click_cooldown()
        self.bot.train_plat_enabled = bool(self.train_plat_var.get())
        self.bot.name_capturable = {
            normalize_name(n) for n in self._get_capturable_from_text()
        }

    def _get_capture_rate(self) -> int:
        try:
            value = int(self.capture_rate_var.get())
        except ValueError:
            value = 50
        value = max(1, min(100, value))
        self.capture_rate_var.set(str(value))
        return value

    def _get_world_click_cooldown(self) -> int:
        try:
            value = int(self.world_click_cooldown_var.get())
        except ValueError:
            value = 30
        value = max(1, min(30, value))
        self.world_click_cooldown_var.set(str(value))
        return value

    def _ui_set_state(self, st: str):
        self.after(0, lambda: self.state_var.set(f"Estado: {st}"))

    def _ui_log(self, msg: str):
        def _append():
            self._log_lines.append(msg)
            if len(self._log_lines) > LOG_MAX_LINES:
                self._log_lines = self._log_lines[-LOG_MAX_LINES:]
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self._write_log_file()
        self.after(0, _append)

    def _ui_set_capture_rate(self, rate: Optional[int]):
        if rate is None:
            text = "OCR: --%"
        else:
            text = f"OCR: {rate}%"
        self.after(0, lambda: self.capture_ocr_label_var.set(text))

    def _refresh_capturable_text(self):
        self.miscrit_text.delete("1.0", "end")
        if self.capturable_miscrits:
            self.miscrit_text.insert("end", "\n".join(self.capturable_miscrits))

    def _update_miscrit_suggestions(self, text: str):
        self.miscrit_suggest_list.delete(0, "end")
        needle = normalize_name(text)
        if not needle:
            return
        source = self.miscrit_catalog
        for name in source:
            if normalize_name(name).startswith(needle):
                self.miscrit_suggest_list.insert("end", name)

    def _on_miscrit_entry_change(self, _event):
        text = self.miscrit_entry.get()
        self._update_miscrit_suggestions(text)

    def _on_miscrit_suggest_select(self, _event):
        sel = self.miscrit_suggest_list.curselection()
        if not sel:
            return
        name = self.miscrit_suggest_list.get(sel[0])
        self.miscrit_entry.delete(0, "end")
        self.miscrit_entry.insert(0, name)

    def _add_capturable_miscrit(self):
        raw = self.miscrit_entry.get().strip()
        if not raw:
            return
        if self.miscrit_catalog:
            catalog_norm = {normalize_name(n) for n in self.miscrit_catalog}
            if normalize_name(raw) not in catalog_norm:
                messagebox.showwarning(
                    "Aviso",
                    "Nombre no encontrado en miscrits.txt",
                )
                return
        current_norm = {normalize_name(n) for n in self._get_capturable_from_text()}
        if normalize_name(raw) in current_norm:
            return
        if self.miscrit_text.index("end-1c") != "1.0":
            self.miscrit_text.insert("end", "\n")
        self.miscrit_text.insert("end", raw)
        self._save_capturable_text()
        self._apply_settings()
        self.miscrit_entry.delete(0, "end")
        self._update_miscrit_suggestions("")

    def _get_capturable_from_text(self) -> list:
        text = self.miscrit_text.get("1.0", "end")
        names = []
        for line in text.splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.append(name)
        return names

    def _save_capturable_text(self):
        names = self._get_capturable_from_text()
        save_names_to_file(MISCRITS_CAPTURABLE_FILE, names)

    def _on_capturable_text_modified(self, _event):
        if self.miscrit_text.edit_modified():
            self.miscrit_text.edit_modified(False)
            self._save_capturable_text()
            self._apply_settings()

    def _select_world_click_template(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen para click WORLD",
            filetypes=[("Imagen", "*.png *.jpg *.jpeg *.bmp"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            self.bot.set_world_click_template(path)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {exc}")
            return
        self.world_click_tpl_var.set(os.path.basename(path))

    def _clear_world_click_template(self):
        self.bot.set_world_click_template(None)
        self.world_click_tpl_var.set("Sin imagen seleccionada")

    def on_start(self):
        try:
            self._apply_settings()
            self.bot.start()
            self.running_var.set("Ejecutando")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_stop(self):
        self.bot.stop()
        self.running_var.set("Parado")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def on_close(self):
        self.bot.stop()
        if self.keyboard_hotkeys_enabled:
            keyboard.unhook_all_hotkeys()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
