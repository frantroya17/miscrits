import time
import json
import cv2
import numpy as np
from mss import mss
import pyautogui

# --- Config ---
with open("config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

THRESH = CFG.get("threshold", 0.88)  # puedes meterlo en tu config
CLICK_DELAY = CFG.get("click_delay", 0.12)
LOOP_DELAY = CFG.get("loop_delay", 0.25)

# Define el rectángulo del juego (x,y,w,h). Al principio lo pones a mano.
# Luego lo automatizamos buscando la ventana si quieres.
GAME_REGION = CFG.get("game_region", {"left": 0, "top": 0, "width": 1920, "height": 1080})

sct = mss()

def grab():
    img = np.array(sct.grab(GAME_REGION))  # BGRA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def find_template(screen_bgr, template_path, threshold=THRESH):
    tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if tpl is None:
        raise FileNotFoundError(f"No existe template: {template_path}")

    res = cv2.matchTemplate(screen_bgr, tpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        h, w = tpl.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        # devuelve coords ABSOLUTAS en pantalla
        abs_x = GAME_REGION["left"] + center_x
        abs_y = GAME_REGION["top"] + center_y
        return (abs_x, abs_y, float(max_val))
    return None

def click(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(CLICK_DELAY)

# --- Ejemplo: si detecta un botón, lo clica ---
def main():
    template_fight_btn = CFG["templates"]["fight_button"]  # ej: "assets/fight_button.png"

    while True:
        screen = grab()
        match = find_template(screen, template_fight_btn)

        if match:
            x, y, conf = match
            print(f"[BOT] Botón pelea encontrado ({conf:.2f}) -> click en {x},{y}")
            click(x, y)

        time.sleep(LOOP_DELAY)

if __name__ == "__main__":
    pyautogui.FAILSAFE = True  # mover ratón esquina sup izq para parar
    main()
