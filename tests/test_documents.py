from tiendahogar.documents import load_documents


def test_knowledge_base_loads_the_five_supplied_policies():
    documents = load_documents()
    assert [(doc.doc_id, doc.title) for doc in documents] == [
        ("doc1_garantia", "Política de garantía"),
        ("doc2_devoluciones", "Política de devoluciones"),
        ("doc3_envios", "Tiempos de envío"),
        ("doc4_reembolsos", "Reembolsos"),
        ("doc5_contacto", "Canales de contacto"),
    ]


def test_policy_text_keeps_accents_and_punctuation():
    by_id = {doc.doc_id: doc.text for doc in load_documents()}
    assert by_id["doc1_garantia"].startswith("Todos los electrodomésticos grandes")
    assert "(“liquidación”)" in by_id["doc2_devoluciones"]
    assert "supervisor humano — el agente" in by_id["doc4_reembolsos"]
    assert "soporte@tiendahogar.example" in by_id["doc5_contacto"]
