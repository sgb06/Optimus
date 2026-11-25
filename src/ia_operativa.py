import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def generar_recomendaciones(
    resumen_clientes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera una siguiente mejor acción comercial de forma explicable,
    combinando alertas operativas y la predicción de compra.
    """
    recomendaciones = resumen_clientes.copy()

    acciones = []
    prioridades = []
    razones = []

    for _, cliente in recomendaciones.iterrows():
        if cliente["coberturas_vencidas"] > 0:
            accion = "Gestionar cobertura vencida"
            prioridad = "Crítica"
            razon = (
                f"Tiene {int(cliente['coberturas_vencidas'])} "
                "cobertura(s) vencida(s)."
            )

        elif cliente["renovaciones_proximas"] > 0:
            accion = "Contactar por renovación próxima"
            prioridad = "Alta"
            razon = (
                f"Tiene {int(cliente['renovaciones_proximas'])} "
                "renovación(es) próxima(s)."
            )

        elif cliente["cliente_frio"]:
            accion = "Recuperar cliente frío"
            prioridad = "Alta"
            razon = (
                f"Lleva {int(cliente['dias_sin_compra'])} "
                "días sin comprar."
            )

        elif cliente["prioridad_compra_30d"]:
            accion = "Contactar por alta probabilidad de compra"
            prioridad = "Media"
            razon = (
                "El modelo estima una probabilidad de compra de "
                f"{cliente['probabilidad_compra_30d'] * 100:.1f}% "
                "en los próximos 30 días."
            )

        elif cliente["cliente_frecuente"]:
            accion = "Mantener relación con cliente frecuente"
            prioridad = "Media"
            razon = (
                f"Registra {int(cliente['transacciones_confirmadas'])} "
                "compras confirmadas."
            )

        else:
            accion = "Seguimiento comercial habitual"
            prioridad = "Baja"
            razon = "No presenta alertas operativas o comerciales."

        acciones.append(accion)
        prioridades.append(prioridad)
        razones.append(razon)

    recomendaciones["siguiente_accion"] = acciones
    recomendaciones["nivel_prioridad"] = prioridades
    recomendaciones["razon_recomendacion"] = razones

    orden_prioridad = {
        "Crítica": 1,
        "Alta": 2,
        "Media": 3,
        "Baja": 4,
    }

    recomendaciones["orden_prioridad"] = (
        recomendaciones["nivel_prioridad"].map(orden_prioridad)
    )

    return recomendaciones.sort_values(
        [
            "orden_prioridad",
            "probabilidad_compra_30d",
        ],
        ascending=[
            True,
            False,
        ],
    ).drop(columns="orden_prioridad")


def detectar_anomalias(
    ventas: pd.DataFrame,
    porcentaje_anomalias: float = 0.03,
) -> pd.DataFrame:
    """
    Detecta ventas atípicas mediante Isolation Forest.

    Evalúa cantidad, precio, descuento, monto COP y moneda.
    El puntaje no indica fraude: indica que la venta requiere revisión.
    """
    datos_modelo = ventas.copy()

    caracteristicas = pd.DataFrame(
        {
            "cantidad": pd.to_numeric(
                datos_modelo["cantidad"],
                errors="coerce",
            ),
            "precio_unitario": pd.to_numeric(
                datos_modelo["precio_unitario"],
                errors="coerce",
            ),
            "descuento_pct": pd.to_numeric(
                datos_modelo["descuento_pct"],
                errors="coerce",
            ),
            "monto_cop_log": np.log1p(
                pd.to_numeric(
                    datos_modelo["monto_cop"],
                    errors="coerce",
                )
            ),
            "es_usd": (
                datos_modelo["moneda"] == "USD"
            ).astype(int),
        }
    )

    caracteristicas = caracteristicas.fillna(
        caracteristicas.median(numeric_only=True)
    )

    modelo = IsolationForest(
        contamination=porcentaje_anomalias,
        random_state=42,
    )

    predicciones = modelo.fit_predict(caracteristicas)

    datos_modelo["es_anomalia"] = predicciones == -1

    datos_modelo["puntaje_anomalia"] = -modelo.score_samples(
        caracteristicas
    )

    limite_monto = datos_modelo["monto_cop"].quantile(0.95)
    limite_cantidad = datos_modelo["cantidad"].quantile(0.95)
    limite_descuento = datos_modelo["descuento_pct"].quantile(0.95)

    motivos = []

    for _, venta in datos_modelo.iterrows():
        razones = []

        if venta["monto_cop"] >= limite_monto:
            razones.append("Monto inusualmente alto")

        if venta["cantidad"] >= limite_cantidad:
            razones.append("Cantidad inusualmente alta")

        if (
            venta["descuento_pct"] >= limite_descuento
            and venta["descuento_pct"] > 0
        ):
            razones.append("Descuento inusual")

        if not razones and venta["es_anomalia"]:
            razones.append("Patrón multivariable atípico")

        motivos.append(
            " | ".join(razones)
            if razones
            else "Sin alerta"
        )

    datos_modelo["motivo_revision"] = motivos

    columnas_resultado = [
        "venta_id",
        "fecha",
        "cliente_id",
        "producto_id",
        "moneda",
        "cantidad",
        "precio_unitario",
        "descuento_pct",
        "monto_cop",
        "puntaje_anomalia",
        "es_anomalia",
        "motivo_revision",
    ]

    return datos_modelo[
        columnas_resultado
    ].sort_values(
        "puntaje_anomalia",
        ascending=False,
    )