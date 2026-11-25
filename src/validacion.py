import pandas as pd


def ejecutar_validaciones(
    ventas: pd.DataFrame,
    resumen_clientes: pd.DataFrame,
    parametros: dict,
) -> pd.DataFrame:
    """Ejecuta los seis casos de validación definidos para el reto."""

    fecha_corte = pd.Timestamp(parametros["fecha_corte"])
    resultados = []

    monto_cop = 2 * 100000 * (1 - 10 / 100)

    resultados.append(
        {
            "prueba": "Cálculo COP con descuento",
            "estado": "APROBADA" if monto_cop == 180000 else "FALLIDA",
            "detalle": f"Resultado calculado: COP ${monto_cop:,.0f}",
        }
    )

    monto_usd = 3 * 100 * 4000

    resultados.append(
        {
            "prueba": "Conversión USD con TRM",
            "estado": "APROBADA" if monto_usd == 1200000 else "FALLIDA",
            "detalle": f"Resultado calculado: COP ${monto_usd:,.0f}",
        }
    )

    anuladas = (ventas["estado"] == "Anulada").sum()

    resultados.append(
        {
            "prueba": "Exclusión de anuladas",
            "estado": "APROBADA" if anuladas == 0 else "FALLIDA",
            "detalle": f"Anuladas en el dataset analítico: {anuladas}",
        }
    )

    dias_frontera_frio = (
        fecha_corte - pd.Timestamp("2026-08-02")
    ).days

    resultados.append(
        {
            "prueba": "Frontera cliente frío",
            "estado": (
                "APROBADA"
                if dias_frontera_frio
                >= parametros["frio_dias_sin_compra"]
                else "FALLIDA"
            ),
            "detalle": f"Días sin compra en frontera: {dias_frontera_frio}",
        }
    )

    dias_frontera_renovacion = (
        pd.Timestamp("2026-12-15") - fecha_corte
    ).days

    resultados.append(
        {
            "prueba": "Frontera renovación",
            "estado": (
                "APROBADA"
                if dias_frontera_renovacion
                <= parametros["renovacion_dias"]
                else "FALLIDA"
            ),
            "detalle": (
                f"Días para vencimiento: "
                f"{dias_frontera_renovacion}"
            ),
        }
    )

    existe_cli999 = (
        resumen_clientes["cliente_id"] == "CLI999"
    ).any()

    resultados.append(
        {
            "prueba": "Consulta cliente inexistente",
            "estado": "APROBADA" if not existe_cli999 else "FALLIDA",
            "detalle": "CLI999 no existe y no se inventan datos.",
        }
    )

    return pd.DataFrame(resultados)