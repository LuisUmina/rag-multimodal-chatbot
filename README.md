# rag-multimodal-chatbot

Chatbot RAG multimodal (texto + imagen) con backend en FastAPI para responder preguntas sobre [rag-challenge.pdf](rag-challenge.pdf) con grounding en evidencia del documento.

## Tabla de contenido

- Descripción
- Características
- Arquitectura (alto nivel)
- Requisitos
- Instalación
- Configuración
- Ejecución local
- Uso
- Estructura del proyecto
- Despliegue
- Documentación adicional
- Licencia

## Descripción

Este proyecto implementa un sistema RAG multimodal orientado a evaluación técnica:

- Procesa texto e imágenes del PDF.
- Indexa contenido semántico con embeddings.
- Recupera contexto relevante para responder preguntas.
- Puede incluir referencia visual (imagen/página) cuando aplica.

## Características

- Backend en FastAPI.
- Pipeline de ingesta para PDF multimodal.
- Recuperación semántica con vector store.
- Endpoint de chat con respuestas basadas en fuentes.
- Soporte para streaming en la interfaz.

## Arquitectura (alto nivel)

1. Ingesta del PDF (texto + imágenes).
2. Enriquecimiento multimodal (captions/metadatos de imagen).
3. Chunking + embeddings.
4. Indexación en FAISS/Chroma.
5. Retrieval top-k y generación de respuesta con LLM.
6. Exposición mediante API FastAPI y frontend de demo.

## Requisitos

- Python 3.10 o superior.
- Cuenta y API key de OpenAI.
- Git.

Opcional (si usarás contenedores):

- Docker + Docker Compose.

## Instalación

1. Clonar repositorio:

```bash
git clone <URL_DEL_REPO>
cd rag-multimodal-chatbot
```

2. Crear y activar entorno virtual:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

1. Crear archivo `.env` a partir de `.env.example` (cuando exista).
2. Definir variables mínimas:

```bash
OPENAI_API_KEY=tu_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
VECTOR_STORE=faiss
```

3. Verificar que el documento [rag-challenge.pdf](rag-challenge.pdf) esté en la raíz del proyecto.

## Ejecución local

Levantar API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva:

- `http://localhost:8000/docs`

Si el frontend está habilitado en un proceso separado (ejemplo Gradio), ejecutarlo en otro terminal:

```bash
python app/ui/gradio_app.py
```

## Uso

Flujo sugerido:

1. Ejecutar ingesta del PDF (endpoint o script de ingesta).
2. Verificar cantidad de páginas/chunks indexados.
3. Hacer preguntas por endpoint de chat o UI.
4. Revisar respuesta + fuentes + evidencia visual.

Ejemplos de preguntas para validar el reto:

- "Explica la arquitectura principal y muestra la imagen relacionada".
- "¿Qué función cumple el componente X en el documento?".
- "Resume la sección más importante en 5 puntos".

## Estructura del proyecto

Estructura objetivo (puede evolucionar durante el desarrollo):

```text
rag-multimodal-chatbot/
├── README.md
├── TECHNICAL_GUIDE.md
├── rag-challenge.pdf
├── LICENSE
├── app/
├── tests/
└── data/
```

## Despliegue

Opciones recomendadas para portafolio:

- Opción simple: una sola app en Hugging Face Spaces (Docker).
- Opción separada: frontend en Vercel + backend FastAPI en proveedor con free tier.

Nota: los free tiers cambian con frecuencia; validar límites vigentes antes del deploy final.

## Documentación adicional

Para la guía técnica detallada de implementación, revisar [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).

## Licencia

Este proyecto se distribuye bajo los términos de [LICENSE](LICENSE).