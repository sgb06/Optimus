import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COLUMNAS_MODELO = [
    "recencia_dias",
    "frecuencia_30d",
    "frecuencia_90d",
    "transacciones_historicas",
    "ingreso_historico_cop",
]


def crear_fotografias_modelo(
    ventas: pd.DataFrame,
    clientes: pd.DataFrame,
    parametros: dict,
) -> pd.DataFrame:
    """Crea fotografías históricas sin usar información futura."""
    fecha_corte = pd.Timestamp(parametros["fecha_corte"])
    horizonte = parametros["horizonte_prediccion_dias"]

    clientes = clientes.copy()
    clientes["fecha_alta"] = pd.to_datetime(clientes["fecha_alta"])

    fecha_inicio = ventas["fecha"].min() + pd.offsets.MonthEnd(2)
    fecha_fin = fecha_corte - pd.Timedelta(days=horizonte)

    periodos = pd.period_range(
        start=fecha_inicio.to_period("M"),
        end=fecha_fin.to_period("M"),
        freq="M",
    )

    filas = []

    for periodo in periodos:
        fecha_snapshot = periodo.to_timestamp(how="end").normalize()
        fecha_objetivo = fecha_snapshot + pd.Timedelta(days=horizonte)

        if fecha_objetivo <= pd.Timestamp("2026-04-30"):
            division = "entrenamiento"
        elif fecha_objetivo <= pd.Timestamp("2026-06-30"):
            division = "validacion"
        elif fecha_objetivo <= pd.Timestamp("2026-09-30"):
            division = "prueba"
        else:
            continue

        clientes_activos = clientes[
            clientes["fecha_alta"] <= fecha_snapshot
        ]["cliente_id"]

        ventas_historicas = ventas[ventas["fecha"] <= fecha_snapshot]
        ventas_30d = ventas_historicas[
            ventas_historicas["fecha"] > fecha_snapshot - pd.Timedelta(days=30)
        ]
        ventas_90d = ventas_historicas[
            ventas_historicas["fecha"] > fecha_snapshot - pd.Timedelta(days=90)
        ]
        ventas_futuras = ventas[
            (ventas["fecha"] > fecha_snapshot)
            & (ventas["fecha"] <= fecha_objetivo)
        ]

        compradores_futuros = set(ventas_futuras["cliente_id"])

        for cliente_id in clientes_activos:
            historial = ventas_historicas[
                ventas_historicas["cliente_id"] == cliente_id
            ]
            historial_30d = ventas_30d[
                ventas_30d["cliente_id"] == cliente_id
            ]
            historial_90d = ventas_90d[
                ventas_90d["cliente_id"] == cliente_id
            ]

            if historial.empty:
                recencia = 999
            else:
                recencia = (fecha_snapshot - historial["fecha"].max()).days

            filas.append(
                {
                    "cliente_id": cliente_id,
                    "fecha_snapshot": fecha_snapshot,
                    "division": division,
                    "recencia_dias": recencia,
                    "frecuencia_30d": len(historial_30d),
                    "frecuencia_90d": len(historial_90d),
                    "transacciones_historicas": len(historial),
                    "ingreso_historico_cop": historial["monto_cop"].sum(),
                    "compra_30d": int(cliente_id in compradores_futuros),
                }
            )

    return pd.DataFrame(filas)


def evaluar_modelo(y_real, y_predicho) -> dict:
    return {
        "precision": round(
            precision_score(y_real, y_predicho, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_real, y_predicho, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_real, y_predicho, zero_division=0), 4
        ),
    }


def entrenar_modelo(
    ventas: pd.DataFrame,
    clientes: pd.DataFrame,
    parametros: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Entrena con datos anteriores a mayo de 2026, calibra con mayo-junio
    y evalúa de julio a septiembre. No usa mezcla aleatoria.
    """
    fotografias = crear_fotografias_modelo(ventas, clientes, parametros)

    entrenamiento = fotografias[
        fotografias["division"] == "entrenamiento"
    ].copy()
    validacion = fotografias[
        fotografias["division"] == "validacion"
    ].copy()
    prueba = fotografias[
        fotografias["division"] == "prueba"
    ].copy()

    modelo = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "clasificador",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    modelo.fit(
        entrenamiento[COLUMNAS_MODELO],
        entrenamiento["compra_30d"],
    )

    prob_validacion = modelo.predict_proba(
        validacion[COLUMNAS_MODELO]
    )[:, 1]

    umbrales = np.arange(0.20, 0.81, 0.05)
    umbral = max(
        umbrales,
        key=lambda valor: f1_score(
            validacion["compra_30d"],
            (prob_validacion >= valor).astype(int),
            zero_division=0,
        ),
    )

    prob_prueba = modelo.predict_proba(prueba[COLUMNAS_MODELO])[:, 1]
    pred_prueba = (prob_prueba >= umbral).astype(int)

    pred_regla_base = (
        prueba["recencia_dias"] <= 30
    ).astype(int)

    fecha_corte = pd.Timestamp(parametros["fecha_corte"])
    clientes_actuales = clientes.copy()
    clientes_actuales["fecha_alta"] = pd.to_datetime(
        clientes_actuales["fecha_alta"]
    )

    filas_actuales = []

    for cliente_id in clientes_actuales["cliente_id"]:
        historial = ventas[ventas["cliente_id"] == cliente_id]
        historial = historial[historial["fecha"] <= fecha_corte]

        historial_30d = historial[
            historial["fecha"] > fecha_corte - pd.Timedelta(days=30)
        ]
        historial_90d = historial[
            historial["fecha"] > fecha_corte - pd.Timedelta(days=90)
        ]

        recencia = (
            (fecha_corte - historial["fecha"].max()).days
            if not historial.empty
            else 999
        )

        filas_actuales.append(
            {
                "cliente_id": cliente_id,
                "recencia_dias": recencia,
                "frecuencia_30d": len(historial_30d),
                "frecuencia_90d": len(historial_90d),
                "transacciones_historicas": len(historial),
                "ingreso_historico_cop": historial["monto_cop"].sum(),
            }
        )

    predicciones = pd.DataFrame(filas_actuales)
    predicciones["probabilidad_compra_30d"] = modelo.predict_proba(
        predicciones[COLUMNAS_MODELO]
    )[:, 1]
    predicciones["prioridad_compra_30d"] = (
        predicciones["probabilidad_compra_30d"] >= umbral
    )

    resultados = {
        "modelo": "Regresión logística",
        "umbral_elegido": round(float(umbral), 2),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_prueba": len(prueba),
        "metricas_prueba_modelo": evaluar_modelo(
            prueba["compra_30d"],
            pred_prueba,
        ),
        "metricas_prueba_regla_base": evaluar_modelo(
            prueba["compra_30d"],
            pred_regla_base,
        ),
    }

    return predicciones, resultados