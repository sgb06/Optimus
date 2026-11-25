from pathlib import Path

RUTA_DATOS = Path(
    r"C:\Users\santi\Downloads\2\Green_Services_Recursos_de_Prueba"
)

ARCHIVOS_REQUERIDOS = [
    "01_Datos/ventas.csv",
    "01_Datos/compras.csv",
    "01_Datos/clientes.csv",
    "01_Datos/productos.csv",
    "01_Datos/proveedores.csv",
    "01_Datos/base_instalada.csv",
    "01_Datos/trm_prueba.csv",
    "02_Documentacion/parametros_prueba.json",
    "05_Validacion/Casos_controlados.json",
]


def validar_ruta_datos() -> Path:
    """Confirma que el paquete de datos tenga los archivos necesarios."""
    faltantes = [
        archivo
        for archivo in ARCHIVOS_REQUERIDOS
        if not (RUTA_DATOS / archivo).exists()
    ]

    if faltantes:
        detalle = "\n- ".join(faltantes)
        raise FileNotFoundError(
            f"No se encontraron estos archivos dentro de {RUTA_DATOS}:\n- {detalle}"
        )

    return RUTA_DATOS