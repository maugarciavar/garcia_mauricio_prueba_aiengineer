"""Order-status tool backed by the mock table supplied with the assessment."""

NOT_FOUND_STATUS = "no encontrado"

# Supplied data, kept as given. "—" means the table has no delivery estimate.
_ORDERS: dict[str, dict[str, str]] = {
    "ORD-1001": {"producto": "Refrigeradora", "estado": "En tránsito", "entrega_estimada": "3 días hábiles"},
    "ORD-1002": {"producto": "Licuadora", "estado": "Entregado", "entrega_estimada": "—"},
    "ORD-1003": {"producto": "Lavadora", "estado": "Procesando", "entrega_estimada": "6 días hábiles"},
    "ORD-1004": {"producto": "Tostadora", "estado": "Cancelado", "entrega_estimada": "—"},
}


def consultar_estado_pedido(order_id: str) -> dict:
    """Return the status of an order, or a clear "no encontrado" result.

    Lookup ignores surrounding whitespace and letter case ("ord-1001" matches).
    Nothing else is guessed: an ID without the "ORD-" prefix is not found.
    """
    normalized = order_id.strip().upper() if isinstance(order_id, str) else ""
    order = _ORDERS.get(normalized)
    if order is None:
        return {
            "encontrado": False,
            "order_id": order_id,
            "estado": NOT_FOUND_STATUS,
            "mensaje": "No existe ningún pedido con ese identificador.",
        }
    return {"encontrado": True, "order_id": normalized, **order}
