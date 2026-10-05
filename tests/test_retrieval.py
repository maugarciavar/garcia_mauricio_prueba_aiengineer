import pytest


def retrieved_ids(retriever, query):
    return [hit.document.doc_id for hit in retriever.search(query)]


@pytest.mark.parametrize(
    "query, expected_doc_id",
    [
        ("¿Cuánto dura la garantía de una lavadora?", "doc1_garantia"),
        ("¿Puedo devolver un producto después de 30 días?", "doc2_devoluciones"),
        ("¿Cuánto tarda el envío a la capital?", "doc3_envios"),
        ("¿Cuánto tarda en procesarse un reembolso?", "doc4_reembolsos"),
        ("¿Cómo contacto a un agente humano?", "doc5_contacto"),
    ],
)
def test_clear_spanish_question_ranks_the_right_policy_first(retriever, query, expected_doc_id):
    assert retrieved_ids(retriever, query)[0] == expected_doc_id


@pytest.mark.parametrize(
    "query, expected_doc_id",
    [
        ("How long is the warranty on a blender?", "doc1_garantia"),
        ("What is your return policy?", "doc2_devoluciones"),
        ("Do you ship internationally?", "doc3_envios"),
        ("When will my refund arrive?", "doc4_reembolsos"),
        ("I want to talk to a person", "doc5_contacto"),
    ],
)
def test_english_question_ranks_the_right_spanish_policy_first(retriever, query, expected_doc_id):
    assert retrieved_ids(retriever, query)[0] == expected_doc_id


@pytest.mark.parametrize(
    "query, expected_doc_id",
    [
        ("mi refri dejó de enfriar a los 8 meses, ¿me lo cubren?", "doc1_garantia"),
        ("ya abrí la caja y lo usé, ¿lo puedo regresar?", "doc2_devoluciones"),
        ("hacen entregas fuera del país?", "doc3_envios"),
        ("¿A qué tarjeta me devuelven el dinero?", "doc4_reembolsos"),
    ],
)
def test_colloquial_paraphrase_still_retrieves_the_right_policy(retriever, query, expected_doc_id):
    assert expected_doc_id in retrieved_ids(retriever, query)


@pytest.mark.parametrize(
    "query",
    [
        "Recomiéndame una receta de pasta",
        "Write me a poem about the sea",
        "What is the capital of France?",
        "Hola",
        "",
    ],
)
def test_unrelated_query_retrieves_nothing(retriever, query):
    assert retriever.search(query) == []


def test_results_are_capped_at_top_k_and_sorted_best_first(retriever):
    hits = retriever.search("¿Puedo devolver un producto después de 30 días?")
    assert 1 <= len(hits) <= retriever.top_k
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= retriever.min_score for score in scores)


def test_score_all_covers_every_document(retriever):
    assert len(retriever.score_all("garantía")) == 5
