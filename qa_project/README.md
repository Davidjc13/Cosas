# Sistema de Q&A con RAG y DeepSeek 🚀

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green)
![LangChain](https://img.shields.io/badge/LangChain-0.0.201+-orange)

Sistema de preguntas y respuestas contextuales utilizando Retrieval-Augmented Generation (RAG) y el modelo DeepSeek como LLM.

## Características Principales ✨
- **Procesamiento de documentos** (TXT, PDF)
- **Búsqueda semántica** con embeddings MPNet
- **Generación de respuestas** contextualizadas
- **API RESTful** con FastAPI
- Documentación automática (Swagger/Redoc)
- Soporte para múltiples documentos
- Escalable con FAISS para almacenamiento vectorial

## Requisitos Previos 📋
- Python 3.10+
- API Key de [DeepSeek](https://deepseek.com/)
- Variables de entorno configuradas (`.env`)

## Instalación ⚙️

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/qa-rag-system.git
cd qa-rag-system

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu API_KEY

# Configuración de Documentos 📂

El sistema está diseñado para trabajar con documentos en formato de texto plano (`.txt`). A continuación, se detalla cómo configurar y organizar los documentos para obtener los mejores resultados.

## Estructura de Directorios

```
documentos/
    └── ia/
        ├── etica_ia.txt
        ├── historia_ia.txt
        └── redes_neuronales.txt
```

## Requisitos de los Documentos
1. **Formato**: Archivos de texto plano (`.txt`)
2. **Codificación**: UTF-8
3. **Tamaño recomendado**: Entre 500 y 5000 palabras por documento
4. **Estructura**: Texto continuo o con párrafos claramente separados

## Ejemplo de Documento
`documentos/ia/etica_ia.txt`

```
Los principales desafíos éticos en IA incluyen:

    Sesgos algorítmicos: Los modelos pueden perpetuar prejuicios presentes en los datos de entrenamiento.

    Privacidad: La recolección y uso de datos personales plantea riesgos significativos.

    Desplazamiento laboral: La automatización podría afectar ciertos empleos.

La UE propuso en 2023 regulaciones específicas para sistemas de IA de alto riesgo.
```


## Cómo Agregar Nuevos Documentos
1. Crea un nuevo archivo `.txt` en el directorio `documentos/`
2. Usa subdirectorios para organizar por temas (opcional)
3. Asegúrate de que el contenido esté bien formateado

## Procesamiento Automático
El sistema:
- Detecta automáticamente nuevos archivos
- Divide el contenido en chunks óptimos
- Genera embeddings para cada segmento
- Indexa los documentos para búsquedas rápidas

## Mejores Prácticas
- **Títulos claros**: Usa nombres descriptivos para los archivos
- **Fuentes confiables**: Asegúrate de usar información verificada
- **Actualización periódica**: Mantén los documentos actualizados
- **Metadatos**: Incluye referencias o fechas en el contenido

## Ejemplo de Uso

En Python:
```python
from core.qa_system import QASystem

qa = QASystem()
result = qa.query("¿Qué regulación propuso la UE?")
print(result['answer'])
```

# 🚀 Uso del Sistema de Q&A con RAG y DeepSeek

Este sistema permite realizar preguntas y obtener respuestas basadas en documentos procesados previamente.

---

## 🏁 Iniciar la API

Para ejecutar la API, usa el siguiente comando (dentro de la carpeta de proyecto):

```bash
uvicorn api.main:app --reload
```
Esto iniciará el servidor en `http://localhost:8000`.

# 🔍 Realizar Consultas

# 📡 Vía cURL

Puedes hacer preguntas al sistema enviando una solicitud POST:
```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "¿Qué regulación propuso la UE?", "max_sources": 2}'
```

# 🌐 Usando Swagger UI

Puedes probar la API directamente desde la documentación interactiva en:
➡️ `http://localhost:8000/docs`

