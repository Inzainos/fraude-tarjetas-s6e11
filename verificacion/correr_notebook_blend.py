#!/usr/bin/env python3
"""Ejecuta un notebook celda por celda (sin jupyter): extrae las celdas
de código y las ejecuta en orden en un mismo namespace, con el cwd en
el directorio del notebook. Uso:
    python3 verificacion/correr_notebook_blend.py [notebook.ipynb]
Por defecto: fraude-tarjetas-blend.ipynb. Imprime el progreso por celda.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "fraude-tarjetas-blend.ipynb")

os.chdir(BASE)
with open(NB, encoding="utf-8") as fh:
    nb = json.load(fh)

ns = {"__name__": "__main__"}
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Ejecutando {len(code_cells)} celdas de código de {NB}", flush=True)
t0 = time.time()
for i, cell in enumerate(code_cells, 1):
    src = "".join(cell["source"])
    print(f"\n===== celda {i}/{len(code_cells)} =====", flush=True)
    t1 = time.time()
    try:
        exec(compile(src, f"<celda {i}>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR en celda {i}: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(f"(celda {i} en {time.time() - t1:.1f}s)", flush=True)
print(f"\nNotebook completo en {time.time() - t0:.1f}s", flush=True)
