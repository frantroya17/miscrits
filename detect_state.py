import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import pyautogui
import cv2
import numpy as np
from mss import mss

MONITOR = 1
SLEEP_SEC = 0.25
THRESH_DEFAULT = 0.86
CLICK_DELAY = 0.15

pyautogui.FAILSAFE = True  # mover ratón a esquina sup-izq para parar

ACTION_COOLDOWN_SEC = 1.0
last_action_time = 0.0

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
    # 1) Popups primero (return inmediato)
    for state in ["CAPTURED_POPUP", "VICTORY"]:
        tpl_img, thr = tpls[state]
        if match(screen_gray, tpl_img) >= thr:
            return state

    # 2) Luego combate
    for state in ["FIGHT_MY_TURN", "FIGHT_WAIT"]:
        tpl_img, thr = tpls[state]
        if match(screen_gray, tpl_img) >= thr:
            return state

    # 3) Mundo al final
    tpl_img, thr = tpls["WORLD"]
    if match(screen_gray, tpl_img) >= thr:
        return "WORLD"

    return None

def main():
    global last_action_time

    templates = {
        "WORLD": Template("tpl/world.png", 0.86),
        "CAPTURED_POPUP": Template("tpl/captured.png", 0.86),
        "FIGHT_MY_TURN": Template("tpl/my_turn.png", 0.86),
        "FIGHT_WAIT": Template("tpl/wait.png", 0.86),
        "VICTORY": Template("tpl/victory.png", 0.86),
    }

    tpls_loaded: Dict[str, Tuple[np.ndarray, float]] = {}
    for k, t in templates.items():
        tpls_loaded[k] = (load_tpl(t.path), t.threshold)

    btn_continue = load_tpl("tpl/btn_continue.png")
    btn_save = load_tpl("tpl/btn_save.png")
    BTN_THR = 0.85

    sct = mss()
    last = None

    while True:
        img = np.array(sct.grab(sct.monitors[MONITOR]))  # BGRA
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        state = detect_state(gray, tpls_loaded)

        # Ignora pantallas intermedias (carga/transición)
        if state is None:
            state = last

        if state != last:
            print(f"[STATE] {state}")
            last = state

        # Cooldown global para evitar clicks repetidos
        now = time.time()
        can_act = (now - last_action_time) >= ACTION_COOLDOWN_SEC

        # Acción 1: Victoria -> Continuar
        if can_act and state == "VICTORY":
            pos = find_center(gray, btn_continue, BTN_THR)
            if pos:
                x, y, conf = pos
                print(f"[ACTION] Click CONTINUAR ({conf:.2f})")
                click_at(x, y)
                last_action_time = time.time()
            else:
                print("[WARN] No encuentro el botón Continuar")

        # Acción 2: Capturado -> Guardar
        elif can_act and state == "CAPTURED_POPUP":
            pos = find_center(gray, btn_save, BTN_THR)
            if pos:
                x, y, conf = pos
                print(f"[ACTION] Click GUARDAR ({conf:.2f})")
                click_at(x, y)
                last_action_time = time.time()
            else:
                print("[WARN] No encuentro el botón Guardar")

        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()
