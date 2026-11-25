import plotly.express as px
import streamlit as st

from src.datos import cargar_parametros, cargar_tablas
from src.finanzas import construir_resumen_financiero
from src.ia_operativa import detectar_anomalias, generar_recomendaciones
from src.indicadores import construir_resumen_clientes
from src.modelo import entrenar_modelo
from src.transformaciones import preparar_compras, preparar_ventas
from src.validacion import ejecutar_validaciones


st.set_page_config(
    page_title="OptimusOps AI",
    page_icon="🌿",
    layout="wide",
)


@st.cache_data
def ejecutar_pipeline():
    """Carga datos, procesa indicadores, IA y validaciones."""
    tablas = cargar_tablas()
    parametros = cargar_parametros()

    ventas_brutas = len(tablas["ventas"])

    ventas = preparar_ventas(tablas)
    compras = preparar_compras(tablas)

    resumen_clientes = construir_resumen_clientes(
        tablas,
        ventas,
        parametros,
    )

    predicciones, resultado_modelo = entrenar_modelo(
        ventas,
        tablas["clientes"],
        parametros,
    )

    resumen_clientes = resumen_clientes.merge(
        predicciones[
            [
                "cliente_id",
                "probabilidad_compra_30d",
                "prioridad_compra_30d",
            ]
        ],
        on="cliente_id",
        how="left",
    )

    recomendaciones = generar_recomendaciones(resumen_clientes)

    anomalias = detectar_anomalias(ventas)

    resumen_financiero, kpis = construir_resumen_financiero(
        ventas,
        compras,
        parametros,
    )

    validaciones = ejecutar_validaciones(
        ventas,
        resumen_clientes,
        parametros,
    )

    return (
        ventas_brutas,
        parametros,
        ventas,
        resumen_clientes,
        recomendaciones,
        anomalias,
        resumen_financiero,
        kpis,
        resultado_modelo,
        validaciones,
    )


(
    ventas_brutas,
    parametros,
    ventas,
    resumen_clientes,
    recomendaciones,
    anomalias,
    resumen_financiero,
    kpis,
    resultado_modelo,
    validaciones,
) = ejecutar_pipeline()


st.title("🌿 OptimusOps AI")

st.caption(
    "Copiloto operativo, predictivo y de detección de anomalías para "
    "datos sintéticos de prueba. No representa cifras reales de "
    "Green Services."
)

st.info(
    f"Fecha de corte: {parametros['fecha_corte']} | "
    f"Ventas confirmadas: {len(ventas)} | "
    f"Ventas anuladas excluidas: {ventas_brutas - len(ventas)}"
)

vista = st.sidebar.radio(
    "Selecciona una vista",
    [
        "Resumen ejecutivo",
        "Operaciones",
        "Comercial",
        "Anomalías IA",
        "Finanzas",
    ],
)


if vista == "Resumen ejecutivo":
    columna_1, columna_2, columna_3, columna_4 = st.columns(4)

    columna_1.metric(
        "Ingresos confirmados",
        f"COP ${kpis['ventas_totales_cop']:,.0f}",
    )

    columna_2.metric(
        "Clientes fríos",
        int(resumen_clientes["cliente_frio"].sum()),
    )

    columna_3.metric(
        "Renovaciones próximas",
        int(resumen_clientes["renovaciones_proximas"].sum()),
    )

    columna_4.metric(
        "Ventas para revisión",
        int(anomalias["es_anomalia"].sum()),
    )

    st.subheader("Clientes priorizados")

    st.dataframe(
        recomendaciones[
            [
                "cliente_id",
                "nombre_cliente",
                "sector",
                "nivel_prioridad",
                "siguiente_accion",
                "probabilidad_compra_30d",
            ]
        ],
        use_container_width=True,
    )


elif vista == "Operaciones":
    st.subheader("Alertas de renovaciones")

    alertas = resumen_clientes[
        resumen_clientes["renovaciones_proximas"] > 0
    ].sort_values(
        "renovaciones_proximas",
        ascending=False,
    )

    st.dataframe(
        alertas[
            [
                "cliente_id",
                "nombre_cliente",
                "unidades_instaladas",
                "coberturas_vencidas",
                "renovaciones_proximas",
            ]
        ],
        use_container_width=True,
    )


elif vista == "Comercial":
    st.subheader("Priorización comercial")

    filtro_frio = st.checkbox("Mostrar solo clientes fríos")
    tabla_comercial = resumen_clientes.copy()

    if filtro_frio:
        tabla_comercial = tabla_comercial[
            tabla_comercial["cliente_frio"]
        ]

    st.dataframe(
        tabla_comercial.sort_values(
            "probabilidad_compra_30d",
            ascending=False,
        )[
            [
                "cliente_id",
                "nombre_cliente",
                "transacciones_confirmadas",
                "dias_sin_compra",
                "cliente_frio",
                "cliente_frecuente",
                "ingreso_cop",
                "probabilidad_compra_30d",
            ]
        ],
        use_container_width=True,
    )

    cliente_id = st.selectbox(
        "Consultar cliente",
        resumen_clientes["cliente_id"].sort_values(),
    )

    cliente = resumen_clientes[
        resumen_clientes["cliente_id"] == cliente_id
    ].iloc[0]

    recomendacion_cliente = recomendaciones[
        recomendaciones["cliente_id"] == cliente_id
    ].iloc[0]

    st.subheader(f"Ficha de {cliente['nombre_cliente']}")

    compras_confirmadas = int(cliente["transacciones_confirmadas"])

    dias_sin_compra = (
        "Sin compras"
        if cliente["sin_historial_compras"]
        else f"{int(cliente['dias_sin_compra'])} días"
    )

    probabilidad = float(cliente["probabilidad_compra_30d"])

    columna_1, columna_2, columna_3, columna_4 = st.columns(4)

    columna_1.metric(
        "Compras confirmadas",
        compras_confirmadas,
    )

    columna_2.metric(
        "Días sin compra",
        dias_sin_compra,
    )

    columna_3.metric(
        "Probabilidad de compra",
        f"{probabilidad * 100:.1f}%",
    )

    columna_4.metric(
        "Renovaciones próximas",
        int(cliente["renovaciones_proximas"]),
    )

    st.progress(
        probabilidad,
        text=(
            "Probabilidad estimada de compra en los próximos 30 días: "
            f"{probabilidad * 100:.1f}%"
        ),
    )

    with st.container(border=True):
        st.markdown("#### 🤖 Recomendación IA para este cliente")

    columna_ia_1, columna_ia_2 = st.columns(2)

    columna_ia_1.metric(
        "Nivel de prioridad",
        recomendacion_cliente["nivel_prioridad"],
    )

    st.markdown(
        f"**Acción sugerida:** "
        f"{recomendacion_cliente['siguiente_accion']}"
    )

    st.caption(
        f"Justificación: "
        f"{recomendacion_cliente['razon_recomendacion']}"
    )

    if cliente["cliente_frio"]:
        st.warning(
            "Cliente frío: lleva 90 días o más sin realizar una compra."
        )
    else:
        st.success(
            "Cliente activo: su última compra está dentro del umbral "
            "de seguimiento."
        )

    if cliente["cliente_frecuente"]:
        st.info(
            "Cliente frecuente: cumple el mínimo de transacciones definido."
        )

elif vista == "Anomalías IA":
    st.subheader("Detección inteligente de ventas atípicas")

    st.caption(
        "Isolation Forest identifica ventas con patrones distintos al "
        "comportamiento histórico. Una alerta requiere revisión; "
        "no confirma fraude o error."
    )

    anomalias_detectadas = anomalias[
        anomalias["es_anomalia"]
    ].copy()

    columna_1, columna_2, columna_3 = st.columns(3)

    columna_1.metric(
        "Ventas analizadas",
        len(anomalias),
    )

    columna_2.metric(
        "Ventas para revisión",
        len(anomalias_detectadas),
    )

    columna_3.metric(
        "Porcentaje marcado",
        f"{len(anomalias_detectadas) / len(anomalias) * 100:.1f}%",
    )

    grafica_anomalias = px.scatter(
        anomalias,
        x="cantidad",
        y="monto_cop",
        color="es_anomalia",
        hover_data=[
            "venta_id",
            "cliente_id",
            "producto_id",
            "motivo_revision",
        ],
        labels={
            "cantidad": "Cantidad vendida",
            "monto_cop": "Monto en COP",
            "es_anomalia": "Requiere revisión",
        },
        title="Ventas normales y ventas atípicas detectadas",
    )

    st.plotly_chart(grafica_anomalias, use_container_width=True)

    st.subheader("Ventas sugeridas para revisión")

    st.dataframe(
        anomalias_detectadas[
            [
                "venta_id",
                "fecha",
                "cliente_id",
                "producto_id",
                "moneda",
                "cantidad",
                "monto_cop",
                "puntaje_anomalia",
                "motivo_revision",
            ]
        ],
        use_container_width=True,
    )


elif vista == "Finanzas":
    st.subheader("Ventas y compras mensuales en COP")

    figura = px.bar(
        resumen_financiero,
        x="mes",
        y=["ventas_cop", "compras_cop"],
        barmode="group",
        labels={
            "value": "Monto en COP",
            "mes": "Mes",
            "variable": "Concepto",
        },
    )

    st.plotly_chart(figura, use_container_width=True)
    st.dataframe(resumen_financiero, use_container_width=True)


elif vista == "Modelo y validación":
    st.subheader("Modelo predictivo de compra a 30 días")

    columna_1, columna_2, columna_3, columna_4 = st.columns(4)

    columna_1.metric(
        "Umbral elegido",
        resultado_modelo["umbral_elegido"],
    )

    columna_2.metric(
        "Precisión del modelo",
        f"{resultado_modelo['metricas_prueba_modelo']['precision'] * 100:.1f}%",
    )

    columna_3.metric(
        "Recall del modelo",
        f"{resultado_modelo['metricas_prueba_modelo']['recall'] * 100:.1f}%",
    )

    columna_4.metric(
        "F1 del modelo",
        f"{resultado_modelo['metricas_prueba_modelo']['f1'] * 100:.1f}%",
    )

    st.subheader("Comparación con la regla base")

    comparacion_modelos = {
        "Métrica": ["Precisión", "Recall", "F1"],
        "Modelo predictivo": [
            resultado_modelo["metricas_prueba_modelo"]["precision"],
            resultado_modelo["metricas_prueba_modelo"]["recall"],
            resultado_modelo["metricas_prueba_modelo"]["f1"],
        ],
        "Regla base": [
            resultado_modelo["metricas_prueba_regla_base"]["precision"],
            resultado_modelo["metricas_prueba_regla_base"]["recall"],
            resultado_modelo["metricas_prueba_regla_base"]["f1"],
        ],
    }

    st.bar_chart(
        comparacion_modelos,
        x="Métrica",
        y=["Modelo predictivo", "Regla base"],
    )

    st.subheader("Casos controlados")
    st.dataframe(validaciones, use_container_width=True)