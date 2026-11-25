import pandas as pd


def construir_resumen_financiero(
    ventas: pd.DataFrame,
    compras: pd.DataFrame,
    parametros: dict,
) -> tuple[pd.DataFrame, dict]:
    """Consolida ventas y compras mensuales hasta la fecha de corte."""

    fecha_corte = pd.Timestamp(parametros["fecha_corte"])

    ventas_corte = ventas[ventas["fecha"] <= fecha_corte].copy()
    compras_corte = compras[compras["fecha"] <= fecha_corte].copy()

    ventas_corte["mes"] = ventas_corte["fecha"].dt.to_period("M").astype(str)
    compras_corte["mes"] = compras_corte["fecha"].dt.to_period("M").astype(str)

    ventas_mensuales = (
        ventas_corte.groupby("mes", as_index=False)
        .agg(
            ventas_cop=("monto_cop", "sum"),
            transacciones=("venta_id", "count"),
        )
    )

    compras_mensuales = (
        compras_corte.groupby("mes", as_index=False)
        .agg(
            compras_cop=("monto_cop", "sum"),
            ordenes_compra=("compra_id", "count"),
        )
    )

    resumen_mensual = (
        ventas_mensuales
        .merge(compras_mensuales, on="mes", how="outer")
        .fillna(0)
        .sort_values("mes")
    )

    resumen_mensual["balance_operativo_cop"] = (
        resumen_mensual["ventas_cop"]
        - resumen_mensual["compras_cop"]
    )

    kpis = {
        "ventas_totales_cop": resumen_mensual["ventas_cop"].sum(),
        "compras_totales_cop": resumen_mensual["compras_cop"].sum(),
        "balance_operativo_cop": resumen_mensual["balance_operativo_cop"].sum(),
    }

    return resumen_mensual, kpis