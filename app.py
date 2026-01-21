import time
import threading
import re
import os
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
import keyboard


MONITOR = 1
SLEEP_SEC = 0.25
THRESH_DEFAULT = 0.86
CLICK_DELAY = 0.15
ACTION_COOLDOWN_SEC = 1.0
OCR_COOLDOWN_SEC = 0.5
RARITY_TPL_THRESHOLD = 0.86
WORLD_CLICK_TPL_THRESHOLD = 0.88
WINDOW_TITLE = "Miscrits"

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
    "tech_left_arrow": (228, 633),
    "tech_right_arrow": (940, 633),
    "tech_slots": [
        (361, 638),
        (529, 634),
        (670, 629),
        (829, 633),
    ],
    "capture_ocr_rect": [(561, 136), (605, 149)],
    "neutral_hover_pos": (30, 30),
}


def get_game_window():
    wins = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not wins:
        return None
    w = wins[0]
    if w.isMinimized:
        w.restore()
        time.sleep(0.2)
    return w


def click_rel(rel_x, rel_y) -> bool:
    w = get_game_window()
    if not w:
        return False
    abs_x = w.left + rel_x
    abs_y = w.top + rel_y
    pyautogui.moveTo(abs_x, abs_y)
    pyautogui.click()
    time.sleep(CLICK_DELAY)
    return True


def move_rel(rel_x, rel_y) -> bool:
    w = get_game_window()
    if not w:
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


def click_at(x, y):
    pyautogui.moveTo(x, y)
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

        # Config (desde UI)
        self.auto_continue = True
        self.auto_save = True
        self.kill_attack_index = 1      # 1..12
        self.capture_attack_index = 1   # 1..12  <-- NUEVO
        self.capture_success_rate = 50  # 1..100 (%)
        self.world_click_cooldown_sec = 30
        self.world_click_template_path: Optional[str] = None
        self.world_click_template: Optional[np.ndarray] = None
        self.last_rarity = None
        self.rarity_capturable = {key: True for _, key in RARITIES}

        # Coordenadas (RELATIVAS)
        self.tech_left_arrow = BOT_COORDS["tech_left_arrow"]
        self.tech_right_arrow = BOT_COORDS["tech_right_arrow"]
        self.tech_slots = BOT_COORDS["tech_slots"]
        self.capture_ocr_rect = BOT_COORDS["capture_ocr_rect"]
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
        self.btn_save = load_tpl("tpl/btn_save.png")
        self.btn_thr = 0.85

        # Templates de rareza
        self.rarity_tpls: Dict[str, Tuple[np.ndarray, float]] = {}
        for _, key in RARITIES:
            tpl_path = f"tpl/rareza_{key}.png"
            alt_path = f"rareza_{key}.png"
            if os.path.exists(tpl_path):
                self.rarity_tpls[key] = (load_tpl(tpl_path), RARITY_TPL_THRESHOLD)
            elif os.path.exists(alt_path):
                self.rarity_tpls[key] = (load_tpl(alt_path), RARITY_TPL_THRESHOLD)
            else:
                self.log(f"[WARN] Falta template de rareza: {tpl_path} (o {alt_path})")

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

    def _detect_rarity(self, screen_gray: np.ndarray) -> Optional[str]:
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

    def _run_loop(self):
        self.log("[BOT] Iniciado")
        try:
            sct = mss()
            monitor = sct.monitors[MONITOR]
            fight_states = {"FIGHT_WAIT", "FIGHT_MY_TURN"}

            while not self.stop_event.is_set():
                img = np.array(sct.grab(monitor))
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

                if can_act and state == "VICTORY" and self.auto_continue:
                    pos = find_center(gray, self.btn_continue, self.btn_thr)
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
                    attack_type = "MATAR"
                    rarity = self._detect_rarity(gray)

                    if rarity and self.rarity_capturable.get(rarity, False):
                        rate = None
                        if (time.time() - self.last_ocr_time) >= OCR_COOLDOWN_SEC:
                            window = get_game_window()
                            if window:
                                rate = self._read_capture_rate(gray, window, monitor)
                                self.set_capture_rate(rate)
                                self.last_ocr_time = time.time()
                            else:
                                self.set_capture_rate(None)
                        else:
                            rate = self.last_capture_rate

                        if rate is not None and rate >= self.capture_success_rate:
                            attack_type = "CAPTURAR"

                    if attack_type == "CAPTURAR":
                        self.log(f"[ACTION] Ataque CAPTURA #{self.capture_attack_index}")
                        self.select_capture_attack(self.capture_attack_index)
                    else:
                        self.log(f"[ACTION] Ataque MATAR #{self.kill_attack_index}")
                        self.select_kill_attack(self.kill_attack_index)

                    self.last_action_time = time.time()

                if state == "WORLD":
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
        self.title("Miscrits Bot (Python)")
        self.geometry("760x520")
        self.resizable(False, False)

        self.state_var = tk.StringVar(value="Estado: -")
        self.running_var = tk.StringVar(value="Parado")

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, textvariable=self.running_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.state_var).pack(anchor="w", pady=(4, 10))

        btns = ttk.Frame(top)
        btns.pack(fill="x")

        self.btn_start = ttk.Button(btns, text="Start", command=self.on_start)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        ttk.Label(top, text="FailSafe: mueve el ratón a la esquina sup-izq para parar").pack(anchor="w", pady=(8, 0))

        opts = ttk.LabelFrame(top, text="Opciones", padding=8)
        opts.pack(fill="x", pady=(10, 0))

        row1 = ttk.Frame(opts)
        row1.pack(fill="x", anchor="w")

        self.auto_continue_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1,
            text="Auto Continuar",
            variable=self.auto_continue_var,
            command=self._apply_settings,
        ).pack(side="left")

        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row1,
            text="Auto Guardar",
            variable=self.auto_save_var,
            command=self._apply_settings,
        ).pack(side="left", padx=10)

        row2 = ttk.Frame(opts)
        row2.pack(fill="x", anchor="w", pady=(6, 0))

        ttk.Label(row2, text="Ataque MATAR (1-12):").pack(side="left", padx=(0, 6))
        self.kill_combo = ttk.Combobox(row2, values=[str(i) for i in range(1, 13)], width=3, state="readonly")
        self.kill_combo.set("1")
        self.kill_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_settings())
        self.kill_combo.pack(side="left")

        ttk.Label(row2, text="Ataque CAPTURAR (1-12):").pack(side="left", padx=(14, 6))
        self.capture_combo = ttk.Combobox(
            row2,
            values=[str(i) for i in range(1, 13)],
            width=3,
            state="readonly",
        )
        self.capture_combo.set("1")
        self.capture_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_settings())
        self.capture_combo.pack(side="left")

        row3 = ttk.Frame(opts)
        row3.pack(fill="x", anchor="w", pady=(6, 0))

        ttk.Label(row3, text="Éxito Captura (% 1-100):").pack(side="left", padx=(0, 6))
        self.capture_rate_var = tk.StringVar(value="50")
        self.capture_rate_spin = ttk.Spinbox(
            row3,
            from_=1,
            to=100,
            width=5,
            textvariable=self.capture_rate_var,
            command=self._apply_settings,
        )
        self.capture_rate_spin.bind("<FocusOut>", lambda e: self._apply_settings())
        self.capture_rate_spin.bind("<Return>", lambda e: self._apply_settings())
        self.capture_rate_spin.pack(side="left")

        self.capture_ocr_label_var = tk.StringVar(value="OCR: --%")
        ttk.Label(row3, textvariable=self.capture_ocr_label_var).pack(side="left", padx=(10, 0))

        row4 = ttk.Frame(opts)
        row4.pack(fill="x", anchor="w", pady=(8, 0))

        ttk.Label(row4, text="Rarezas atrapables:").pack(side="left", padx=(0, 10))
        self.rarity_images = self._load_rarity_images()
        self.rarity_capturable_vars = {}
        for label, key in RARITIES:
            rarity_frame = ttk.Frame(row4)
            rarity_frame.pack(side="left", padx=(0, 10))
            img = self.rarity_images.get(key)
            if img:
                ttk.Label(rarity_frame, image=img).pack()
            else:
                ttk.Label(rarity_frame, text=label).pack()
            var = tk.BooleanVar(value=True)
            self.rarity_capturable_vars[key] = var
            ttk.Checkbutton(
                rarity_frame,
                text="Atrapable",
                variable=var,
                command=self._apply_settings,
            ).pack()

        row5 = ttk.Frame(opts)
        row5.pack(fill="x", anchor="w", pady=(8, 0))

        ttk.Label(row5, text="Click WORLD (imagen):").pack(side="left", padx=(0, 6))
        self.world_click_tpl_var = tk.StringVar(value="Sin imagen seleccionada")
        ttk.Label(row5, textvariable=self.world_click_tpl_var).pack(side="left")
        ttk.Button(
            row5,
            text="Seleccionar imagen",
            command=self._select_world_click_template,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            row5,
            text="Quitar",
            command=self._clear_world_click_template,
        ).pack(side="left", padx=(6, 0))

        ttk.Label(row5, text="Cooldown WORLD (1-30s):").pack(side="left", padx=(14, 6))
        self.world_click_cooldown_var = tk.StringVar(value="30")
        self.world_click_cooldown_spin = ttk.Spinbox(
            row5,
            from_=1,
            to=30,
            width=4,
            textvariable=self.world_click_cooldown_var,
            command=self._apply_settings,
        )
        self.world_click_cooldown_spin.bind("<FocusOut>", lambda e: self._apply_settings())
        self.world_click_cooldown_spin.bind("<Return>", lambda e: self._apply_settings())
        self.world_click_cooldown_spin.pack(side="left")

        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        self.log_box = tk.Text(mid, height=14, wrap="word")
        self.log_box.pack(fill="both", expand=True)

        self.bot = BotRunner(
            on_state=self._ui_set_state,
            on_log=self._ui_log,
            on_capture_rate=self._ui_set_capture_rate,
        )

        self.keyboard_hotkeys_enabled = False
        try:
            keyboard.add_hotkey("f10", self.on_start)
            keyboard.add_hotkey("esc", self.on_close)
            self.keyboard_hotkeys_enabled = True
        except Exception as exc:
            self._ui_log(f"[WARN] Hotkeys desactivados: {exc}")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_settings(self):
        self.bot.auto_continue = bool(self.auto_continue_var.get())
        self.bot.auto_save = bool(self.auto_save_var.get())
        self.bot.kill_attack_index = int(self.kill_combo.get())
        self.bot.capture_attack_index = int(self.capture_combo.get())
        self.bot.capture_success_rate = self._get_capture_rate()
        self.bot.world_click_cooldown_sec = self._get_world_click_cooldown()
        self.bot.rarity_capturable = {
            key: bool(var.get()) for key, var in self.rarity_capturable_vars.items()
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
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
        self.after(0, _append)

    def _ui_set_capture_rate(self, rate: Optional[int]):
        if rate is None:
            text = "OCR: --%"
        else:
            text = f"OCR: {rate}%"
        self.after(0, lambda: self.capture_ocr_label_var.set(text))

    def _load_rarity_images(self) -> Dict[str, tk.PhotoImage]:
        images: Dict[str, tk.PhotoImage] = {}
        for _, key in RARITIES:
            candidates = [
                f"tpl/rareza_{key}.png",
                f"rareza_{key}.png",
            ]
            for path in candidates:
                if os.path.exists(path):
                    images[key] = tk.PhotoImage(file=path)
                    break
        return images

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
