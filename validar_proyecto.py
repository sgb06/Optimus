from src.datos import cargar_parametros, cargar_tablas
from src.indicadores import construir_resumen_clientes
from src.modelo import entrenar_modelo
from src.transformaciones import preparar_compras, preparar_ventas
from src.finanzas import construir_resumen_financiero
from src.validacion import ejecutar_validaciones


def main():
    tablas = cargar_tablas()
    parametros = cargar_parametros()

    ventas = preparar_ventas(tablas)
    compras = preparar_compras(tablas)

    resumen_clientes = construir_resumen_clientes(
        tablas,
        ventas,
        parametros,
    )

    predicciones, resultados_modelo = entrenar_modelo(
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

    resumen_financiero, kpis_financieros = construir_resumen_financiero(
        ventas,
        compras,
        parametros,
    )

    validaciones = ejecutar_validaciones(
        ventas,
        resumen_clientes,
        parametros,
    )

    print("\n=== KPIs FINANCIEROS ===")
    print(kpis_financieros)

    print("\n=== MODELO PREDICTIVO ===")
    print(resultados_modelo)

    print("\n=== VALIDACIONES ===")
    print(validaciones.to_string(index=False))

    if not (validaciones["estado"] == "APROBADA").all():
        raise RuntimeError("Una o más validaciones fallaron.")

    print("\nProyecto validado correctamente.")


if __name__ == "__main__":
    main()