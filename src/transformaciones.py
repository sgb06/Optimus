import numpy as np
import pandas as pd


def preparar_ventas(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Limpia ventas, excluye anuladas y convierte ventas USD a COP
    usando la TRM de la misma fecha.
    """
    ventas = tablas["ventas"].copy()
    trm = tablas["trm"][["fecha", "cop_por_usd"]].copy()

    ventas["fecha"] = pd.to_datetime(ventas["fecha"])
    trm["fecha"] = pd.to_datetime(trm["fecha"])

    ventas["cantidad"] = pd.to_numeric(ventas["cantidad"])
    ventas["precio_unitario"] = pd.to_numeric(ventas["precio_unitario"])
    ventas["descuento_pct"] = pd.to_numeric(ventas["descuento_pct"])

    ventas = ventas[ventas["estado"] == "Confirmada"].copy()

    ventas["monto_neto_origen"] = (
        ventas["cantidad"]
        * ventas["precio_unitario"]
        * (1 - ventas["descuento_pct"] / 100)
    )

    ventas = ventas.merge(
        trm,
        on="fecha",
        how="left",
        validate="many_to_one",
    )

    ventas_usd_sin_trm = (
        (ventas["moneda"] == "USD")
        & ventas["cop_por_usd"].isna()
    )

    if ventas_usd_sin_trm.any():
        fechas = ventas.loc[
            ventas_usd_sin_trm, "fecha"
        ].dt.strftime("%Y-%m-%d")

        raise ValueError(
            "Hay ventas USD sin TRM para estas fechas: "
            + ", ".join(fechas.unique())
        )

    ventas["monto_cop"] = np.where(
        ventas["moneda"] == "USD",
        ventas["monto_neto_origen"] * ventas["cop_por_usd"],
        ventas["monto_neto_origen"],
    )

    return ventas


def preparar_compras(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convierte compras a COP usando la TRM correspondiente a su fecha.
    """
    compras = tablas["compras"].copy()
    trm = tablas["trm"][["fecha", "cop_por_usd"]].copy()

    compras["fecha"] = pd.to_datetime(compras["fecha"])
    trm["fecha"] = pd.to_datetime(trm["fecha"])

    compras["cantidad"] = pd.to_numeric(compras["cantidad"])
    compras["costo_unitario"] = pd.to_numeric(compras["costo_unitario"])

    compras["monto_origen"] = (
        compras["cantidad"] * compras["costo_unitario"]
    )

    compras = compras.merge(
        trm,
        on="fecha",
        how="left",
        validate="many_to_one",
    )

    compras_usd_sin_trm = (
        (compras["moneda"] == "USD")
        & compras["cop_por_usd"].isna()
    )

    if compras_usd_sin_trm.any():
        raise ValueError("Hay compras USD sin TRM para su fecha.")

    compras["monto_cop"] = np.where(
        compras["moneda"] == "USD",
        compras["monto_origen"] * compras["cop_por_usd"],
        compras["monto_origen"],
    )

    return compras