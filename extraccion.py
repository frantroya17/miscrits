import requests

API = "https://miscrits.fandom.com/api.php"
CATEGORY = "Category:Miscrits"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (MiscritsBot/1.0)"
})

names = set()
cmcontinue = None

while True:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": CATEGORY,
        "cmlimit": "500",   # máximo permitido normalmente
        "format": "json"
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    r = session.get(API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    members = data.get("query", {}).get("categorymembers", [])
    for m in members:
        title = m.get("title", "").strip()
        if title:
            # En categorías, el title es el nombre de la página
            names.add(title)

    cmcontinue = data.get("continue", {}).get("cmcontinue")
    print(f"Recibidos: {len(members)} | Total acumulado: {len(names)}")

    if not cmcontinue:
        break

out_file = "miscrits_list.txt"
with open(out_file, "w", encoding="utf-8") as f:
    for name in sorted(names):
        f.write(name + "\n")

print("\nOK. Total:", len(names))
print("Archivo:", out_file)
