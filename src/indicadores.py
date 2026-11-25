import pandas as pd


def construir_resumen_clientes(
    tablas: dict[str, pd.DataFrame],
    ventas: pd.DataFrame,
    parametros: dict,
) -> pd.DataFrame:
    """Genera indicadores comerciales y operativos por cliente."""

    fecha_corte = pd.Timestamp(parametros["fecha_corte"])
    dias_frio = parametros["frio_dias_sin_compra"]
    min_transacciones = parametros["frecuente_min_transacciones"]
    dias_renovacion = parametros["renovacion_dias"]

    clientes = tablas["clientes"].copy()
    clientes = clientes.rename(
        columns={
            "nombre_prueba": "nombre_cliente",
            "sector_prueba": "sector",
            "ciudad_prueba": "ciudad",
        }
    )

    # Solo consideramos ventas hasta la fecha de corte.
    ventas_corte = ventas[ventas["fecha"] <= fecha_corte].copy()

    resumen_ventas = (
        ventas_corte.groupby("cliente_id", as_index=False)
        .agg(
            transacciones_confirmadas=("venta_id", "count"),
            ultima_compra=("fecha", "max"),
            ingreso_cop=("monto_cop", "sum"),
            productos_distintos=("producto_id", "nunique"),
        )
    )

    # Preparamos las instalaciones para alertas de renovación.
    instalaciones = tablas["base_instalada"].copy()
    instalaciones["fecha_inicio"] = pd.to_datetime(instalaciones["fecha_inicio"])
    instalaciones["fecha_fin"] = pd.to_datetime(instalaciones["fecha_fin"])
    instalaciones["unidades"] = pd.to_numeric(instalaciones["unidades"])

    instalaciones = instalaciones[
        instalaciones["fecha_inicio"] <= fecha_corte
    ].copy()

    instalaciones["dias_para_vencer"] = (
        instalaciones["fecha_fin"] - fecha_corte
    ).dt.days

    resumen_instalaciones = (
        instalaciones.groupby("cliente_id", as_index=False)
        .agg(
            unidades_instaladas=("unidades", "sum"),
            coberturas_vencidas=(
                "dias_para_vencer",
                lambda serie: (serie < 0).sum(),
            ),
            renovaciones_proximas=(
                "dias_para_vencer",
                lambda serie: ((serie >= 0) & (serie <= dias_renovacion)).sum(),
            ),
        )
    )

    resumen = (
        clientes[
            ["cliente_id", "nombre_cliente", "sector", "ciudad"]
        ]
        .merge(resumen_ventas, on="cliente_id", how="left")
        .merge(resumen_instalaciones, on="cliente_id", how="left")
    )

    columnas_numericas = [
        "transacciones_confirmadas",
        "ingreso_cop",
        "productos_distintos",
        "unidades_instaladas",
        "coberturas_vencidas",
        "renovaciones_proximas",
    ]

    resumen[columnas_numericas] = resumen[columnas_numericas].fillna(0)

    resumen["dias_sin_compra"] = (
        fecha_corte - resumen["ultima_compra"]
    ).dt.days

    # Un cliente sin compras queda como "sin historial", no como cliente frío.
    resumen["sin_historial_compras"] = resumen["ultima_compra"].isna()
    resumen["cliente_frio"] = (
        resumen["dias_sin_compra"].notna()
        & (resumen["dias_sin_compra"] >= dias_frio)
    )
    resumen["cliente_frecuente"] = (
        resumen["transacciones_confirmadas"] >= min_transacciones
    )

    return resumen