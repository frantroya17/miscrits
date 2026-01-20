import time
import pyautogui
import keyboard
import pygetwindow as gw

WINDOW_TITLE = "Miscrits"

def get_window():
    wins = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not wins:
        return None
    w = wins[0]
    if w.isMinimized:
        w.restore()
        time.sleep(0.2)
    return w

print("=== CAPTURADOR RELATIVO A VENTANA ===")
print(f"Ventana objetivo: '{WINDOW_TITLE}'")
print("Pulsa ESPACIO para capturar (REL_X, REL_Y)")
print("Pulsa ESC para salir\n")
time.sleep(0.5)

last_time = 0.0

while True:
    if keyboard.is_pressed("esc"):
        print("Saliendo...")
        break

    if keyboard.is_pressed("space"):
        now = time.time()
        if now - last_time < 0.35:
            time.sleep(0.05)
            continue
        last_time = now

        w = get_window()
        if not w:
            print("No encuentro la ventana 'Miscrits'. Ábrela o revisa el título.")
            continue

        abs_x, abs_y = pyautogui.position()
        rel_x = abs_x - w.left
        rel_y = abs_y - w.top

        print(f"ABS=({abs_x},{abs_y})  WIN_LEFT_TOP=({w.left},{w.top})  REL=({rel_x},{rel_y})")

    time.sleep(0.03)
