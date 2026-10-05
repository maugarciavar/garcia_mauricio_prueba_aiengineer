import inspect

import pytest

from tiendahogar.orders import NOT_FOUND_STATUS, consultar_estado_pedido


def test_signature_matches_the_specification():
    signature = inspect.signature(consultar_estado_pedido)
    assert list(signature.parameters) == ["order_id"]
    assert signature.parameters["order_id"].annotation is str
    assert signature.return_annotation is dict


@pytest.mark.parametrize(
    "order_id, producto, estado, entrega_estimada",
    [
        ("ORD-1001", "Refrigeradora", "En tránsito", "3 días hábiles"),
        ("ORD-1002", "Licuadora", "Entregado", "—"),
        ("ORD-1003", "Lavadora", "Procesando", "6 días hábiles"),
        ("ORD-1004", "Tostadora", "Cancelado", "—"),
    ],
)
def test_valid_order_returns_the_supplied_row(order_id, producto, estado, entrega_estimada):
    assert consultar_estado_pedido(order_id) == {
        "encontrado": True,
        "order_id": order_id,
        "producto": producto,
        "estado": estado,
        "entrega_estimada": entrega_estimada,
    }


@pytest.mark.parametrize("order_id", ["ORD-9999", "1001", "ORD-10011", "ORD1001", "", "   "])
def test_unknown_order_is_reported_as_not_found_without_invented_data(order_id):
    result = consultar_estado_pedido(order_id)
    assert result["encontrado"] is False
    assert result["estado"] == NOT_FOUND_STATUS == "no encontrado"
    assert result["order_id"] == order_id
    assert "producto" not in result
    assert "entrega_estimada" not in result


@pytest.mark.parametrize("order_id", ["ord-1001", "  ORD-1001  ", "Ord-1001\n"])
def test_lookup_ignores_case_and_surrounding_whitespace(order_id):
    result = consultar_estado_pedido(order_id)
    assert result["encontrado"] is True
    assert result["order_id"] == "ORD-1001"
    assert result["producto"] == "Refrigeradora"


@pytest.mark.parametrize("order_id", [None, 1001])
def test_non_string_input_is_not_found_rather_than_an_error(order_id):
    assert consultar_estado_pedido(order_id)["estado"] == NOT_FOUND_STATUS


def test_result_is_a_copy_so_callers_cannot_corrupt_the_table():
    consultar_estado_pedido("ORD-1001")["estado"] = "Entregado"
    assert consultar_estado_pedido("ORD-1001")["estado"] == "En tránsito"
