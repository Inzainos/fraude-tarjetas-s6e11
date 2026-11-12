#!/usr/bin/env python3
"""Ejecuta fraude-tarjetas.ipynb celda por celda (sin jupyter):
extrae las celdas de código y las ejecuta en orden en un mismo
namespace, con el cwd en el directorio del notebook. Imprime el
progreso y el resultado de la verificación final del notebook.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
NB = os.path.join(BASE, "fraude-tarjetas.ipynb")

os.chdir(BASE)
with open(NB, encoding="utf-8") as fh:
    nb = json.load(fh)

ns = {"__name__": "__main__"}
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Ejecutando {len(code_cells)} celdas de código de {NB}")
t0 = time.time()
for i, cell in enumerate(code_cells, 1):
    src = "".join(cell["source"])
    print(f"\n===== celda {i}/{len(code_cells)} =====")
    t1 = time.time()
    try:
        exec(compile(src, f"<celda {i}>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR en celda {i}: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"(celda {i} en {time.time() - t1:.1f}s)")
print(f"\nNotebook completo en {time.time() - t0:.1f}s")
