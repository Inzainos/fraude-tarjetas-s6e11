#!/usr/bin/env python3
"""Verifica que el entorno coincide con requirements.txt y deja un log.

Uso:
    python3 verificar_entorno.py                      # usa ./requirements.txt
    python3 verificar_entorno.py --requirements ruta  # otro archivo
    python3 verificar_entorno.py --log verificacion/entorno.log

Lee los pines "paquete==version" de requirements.txt, compara con las
versiones instaladas (importlib.metadata) y con el Python de referencia.
Sale con codigo 0 si todo coincide y 1 si hay algun paquete ausente o con
otra version. No instala ni modifica nada.
"""
import argparse
import logging
import os
import re
import sys
from importlib import metadata

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_REFERENCIA = (3, 12)
log = logging.getLogger("entorno")


def leer_pines(ruta):
    """Devuelve {paquete: version} para las lineas 'paquete==version'."""
    pines = {}
    with open(ruta, encoding="utf-8") as fh:
        for n, linea in enumerate(fh, 1):
            txt = linea.split("#", 1)[0].strip()
            if not txt:
                continue
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)", txt)
            if not m:
                log.info("linea %d ignorada (no es un pin exacto): %s", n, txt)
                continue
            pines[m.group(1)] = m.group(2)
    return pines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default=os.path.join(HERE, "requirements.txt"))
    ap.add_argument("--log", default=os.path.join(HERE, "verificacion", "entorno.log"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(a.log, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])

    ok = True
    v = sys.version_info
    log.info("python %d.%d.%d (%s)", v.major, v.minor, v.micro, sys.executable)
    if (v.major, v.minor) != PYTHON_REFERENCIA:
        log.info("AVISO: la referencia es Python %d.%d; con otra version los pines "
                 "pueden no instalarse o dar resultados distintos",
                 *PYTHON_REFERENCIA)

    pines = leer_pines(a.requirements)
    if not pines:
        log.info("ERROR: %s no tiene pines exactos", a.requirements)
        return 1
    for paquete, esperada in sorted(pines.items()):
        try:
            instalada = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            log.info("FALTA  %-14s esperado %s", paquete, esperada)
            ok = False
            continue
        if instalada == esperada:
            log.info("OK     %-14s %s", paquete, instalada)
        else:
            log.info("DIFIERE %-14s instalado %s, esperado %s", paquete, instalada, esperada)
            ok = False

    # Prueba de importacion real (un wheel roto puede estar "instalado" y no importar).
    for modulo in ("numpy", "pandas", "scipy", "sklearn", "lightgbm"):
        try:
            __import__(modulo)
            log.info("import %-9s OK", modulo)
        except Exception as e:  # noqa: BLE001
            log.info("import %-9s FALLA: %s", modulo, e)
            ok = False

    log.info("RESULTADO: %s", "entorno coincide con requirements.txt" if ok
             else "entorno NO coincide (ver lineas FALTA/DIFIERE/FALLA)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
