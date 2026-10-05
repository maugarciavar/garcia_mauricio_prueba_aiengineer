# Agente de soporte TiendaHogar

Agente mínimo de soporte al cliente para la empresa ficticia de electrodomésticos TiendaHogar. El agente:

- responde preguntas sobre garantía, devoluciones, envíos y reembolsos usando recuperación (RAG) sobre los cinco documentos de política entregados;
- consulta el estado de un pedido con la tool `consultar_estado_pedido(order_id: str) -> dict`;
- escala los casos que las políticas indican que no debe manejar (reembolsos mayores a $500, quejas sobre el trato de un empleado, disputas de facturación y temas legales);
- indica que no tiene la información cuando esta no está en los documentos ni en la tabla de pedidos.

Las decisiones de diseño, los trade-offs y las limitaciones están en [SUBMISSION.md](SUBMISSION.md).

## Requisitos

- Python 3.11 o superior (desarrollado y probado con Python 3.11.9 en Windows).
- Una API key de OpenAI. Solo se necesita para ejecutar el agente y la evaluación en vivo; **no** se necesita para las pruebas unitarias.
- Acceso a internet en la primera ejecución: el modelo local de embeddings (aproximadamente 0.64 GB) se descarga una vez desde Hugging Face y queda en caché. No se requiere cuenta de Hugging Face.

## Instalación

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Después, abre `.env` y asigna tu clave a `OPENAI_API_KEY`. Git ignora ese archivo.

En Windows, si `pip install` falla con un error de rutas largas (*Long Path*), clona el repositorio en una ruta corta, por ejemplo `C:\th`, y repite la instalación.

### Variables de entorno

| Variable | Obligatoria | Valor por defecto | Uso |
|---|---|---|---|
| `OPENAI_API_KEY` | Para ejecutar el agente | ninguno | API key de OpenAI |
| `OPENAI_MODEL` | No | `gpt-6-luna` | Modelo que genera las respuestas |
| `MAX_TOOL_ROUNDS` | No | `3` | Máximo de rondas de consulta de pedidos por mensaje |
| `EMBEDDING_MODEL` | No | `jinaai/jina-embeddings-v2-base-es` | Modelo de FastEmbed para la recuperación |
| `RAG_TOP_K` | No | `3` | Máximo de políticas recuperadas por mensaje |
| `RAG_MIN_SCORE` | No | `0.20` | Similitud coseno mínima para usar una política |

## Ejecutar el agente

Una sola pregunta:

```bash
python -m tiendahogar.cli "¿Cuánto dura la garantía de una lavadora?"
```

Conversación interactiva (escribe `salir` para terminar):

```bash
python -m tiendahogar.cli
```

Con `--debug`, en cualquiera de las dos formas, se muestra cómo se produjo cada respuesta: la ruta, las políticas recuperadas con su puntaje y las llamadas a la tool.

```bash
python -m tiendahogar.cli --debug "¿Dónde está mi pedido ORD-1001?"
```

### Ejemplo de uso

Respuestas reales del agente con `gpt-6-luna`. Las genera un modelo de lenguaje, así que la redacción puede variar entre ejecuciones.

| Pregunta | Respuesta del agente |
|---|---|
| ¿Cuánto dura la garantía de una lavadora? | La garantía de una lavadora dura 12 meses desde la fecha de compra y cubre defectos de fábrica, no daños por mal uso. |
| ¿Dónde está mi pedido ORD-1001? | Tu pedido ORD-1001, una refrigeradora, está en tránsito. La entrega estimada es en 3 días hábiles. |
| Quiero un reembolso de $800 por mi refrigeradora | Los reembolsos mayores a $500 requieren la aprobación de un supervisor humano, por lo que no puedo aprobar ni procesar esta solicitud. |
| ¿Cuál es el horario de la tienda? | No tengo información sobre el horario de la tienda. |

### Uso desde Python

```python
from dotenv import load_dotenv
from tiendahogar.agent import build_agent

load_dotenv()
agent = build_agent()
response = agent.respond("¿Dónde está mi pedido ORD-1001?")
print(response.text)        # la respuesta
print(response.route)       # "answered", "escalated" o "fallback"
print(response.tool_calls)  # consultas de pedidos realizadas
```

La tool también se puede llamar directamente:

```python
from tiendahogar.orders import consultar_estado_pedido

consultar_estado_pedido("ORD-1001")
# {'encontrado': True, 'order_id': 'ORD-1001', 'producto': 'Refrigeradora',
#  'estado': 'En tránsito', 'entrega_estimada': '3 días hábiles'}
consultar_estado_pedido("ORD-9999")
# {'encontrado': False, 'order_id': 'ORD-9999', 'estado': 'no encontrado', 'mensaje': '...'}
```

## Ejecutar las pruebas

Pruebas unitarias. No requieren API key ni hacen llamadas a un LLM; las pruebas de recuperación cargan el modelo local de embeddings:

```bash
pytest
```

`pytest tests/` es equivalente.

Evaluación en vivo contra el modelo real (requiere `OPENAI_API_KEY`). Son 58 escenarios y unas 60 llamadas al modelo:

```bash
pytest -m live
```

Con `-rA` se imprime cada pregunta y su respuesta.

## Estructura del proyecto

```
src/tiendahogar/
  policies/        los cinco documentos de política, sin modificar (.txt)
  documents.py     carga las políticas
  orders.py        tabla de pedidos y consultar_estado_pedido
  retrieval.py     embeddings + similitud coseno, top-k y puntaje mínimo
  guardrails.py    guardrail en código: reembolsos mayores a $500 (regla aritmética)
  prompts.py       prompt de sistema: reglas de escalamiento (trato de empleados,
                   facturación, temas legales, reembolsos), fundamentación y contexto
  agent.py         guardrail -> recuperación -> LLM con ciclo de tools acotado
  cli.py           interfaz de línea de comandos
  config.py        configuración desde variables de entorno
tests/
  test_*.py        pruebas unitarias (sin conexión al LLM)
  live/            escenarios y ejecución de la evaluación en vivo
```

El código, los comentarios y los nombres de las pruebas están en inglés; los documentos de política y los datos de pedidos se conservan en español, tal como fueron entregados.
