import json
from pathlib import Path

import pandas as pd

from src.configuracion import validar_ruta_datos


ARCHIVOS_DATOS = {
    "ventas": "01_Datos/ventas.csv",
    "compras": "01_Datos/compras.csv",
    "clientes": "01_Datos/clientes.csv",
    "productos": "01_Datos/productos.csv",
    "proveedores": "01_Datos/proveedores.csv",
    "base_instalada": "01_Datos/base_instalada.csv",
    "trm": "01_Datos/trm_prueba.csv",
}


def cargar_parametros() -> dict:
    """Lee los umbrales definidos en el paquete de prueba."""
    ruta = validar_ruta_datos() / "02_Documentacion/parametros_prueba.json"

    with ruta.open(encoding="utf-8-sig") as archivo:
        return json.load(archivo)


def cargar_tablas() -> dict[str, pd.DataFrame]:
    """Carga cada CSV del paquete en un DataFrame de pandas."""
    raiz = validar_ruta_datos()

    return {
        nombre: pd.read_csv(raiz / ruta_relativa, encoding="utf-8-sig")
        for nombre, ruta_relativa in ARCHIVOS_DATOS.items()
    }