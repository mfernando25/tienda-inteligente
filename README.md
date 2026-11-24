# Tienda inteligente

Proyecto académico de la asignatura **Inteligencia de Negocios**.

El proyecto integra Machine Learning, visión por computadora, procesamiento de lenguaje natural y sistemas conversacionales a través de un caso de supermercado.

## Componentes

### Generador de anaqueles
Aplicación web local 2D con un anaquel de 4 filas × 8 columnas, ocho productos ilustrados, escenarios controlados, edición manual y exportaciones PNG, JSON y CSV.

Instrucciones: [shelf-generator/README.md](shelf-generator/README.md).

### Visión por computadora y Machine Learning
Incluye rectificación de perspectiva, preparación del dataset, entrenamiento HOG + SVM e inferencia y evaluación sobre cinco fotografías nuevas con etiquetas de referencia.

Instrucciones técnicas: [cv-lab/README.md](cv-lab/README.md).

### Procesamiento de lenguaje natural
Contiene un dataset de 72 comentarios simulados en español, exploración de texto con NLTK y clasificación de sentimientos mediante un Transformer preentrenado para español, evaluado contra etiquetas de referencia.

Instrucciones: [nlp-lab/README.md](nlp-lab/README.md).

### Sistemas conversacionales
Chatbot local que aprende a clasificar intenciones mediante TF-IDF y Regresión Logística, y consulta el inventario visual y el resumen de sentimientos.

Instrucciones: [conversational-lab/README.md](conversational-lab/README.md).

## Guías de laboratorio

- [Sesión 1: Machine Learning y Visión por computadora](labs/GUIA_PRACTICA_INTEGRACION_ML_VISION_ANAQUEL.md).
- [Sesión 2: Procesamiento de lenguaje natural y Sistemas conversacionales](labs/GUIA_PRACTICA_PNL_SISTEMAS_CONVERSACIONALES.md).

## Pautas de continuidad de la Unidad 4

[Decisiones del proyecto, iteraciones y escenarios del examen](docs/PAUTAS_CONTINUIDAD_UNIDAD4.md).

## Estructura

- `shelf-generator/`: generador de anaqueles.
- `cv-lab/`: scripts OpenCV, preparación del dataset, entrenamiento e inferencia/evaluación.
- `nlp-lab/`: dataset de comentarios y análisis de sentimientos.
- `conversational-lab/`: asistente conversacional local.
- `docs/`: arquitectura y decisiones del proyecto.
- `labs/`: guías prácticas independientes por sesión.
- `exam/`: recursos de evaluación.

## Requisitos y restricciones

- Herramientas gratuitas y preferentemente open source.
- Ejecución local siempre que sea posible.
- Sin dependencia de servicios externos de pago.
- Instrucciones reproducibles para Windows.
- La Sesión 2 utiliza PySentimiento, NLTK y scikit-learn; los modelos se preparan antes de la práctica.

## Estado

- **Generador:** prototipo funcional.
- **Sesión 1:** entrenamiento HOG + SVM y evaluación con cinco fotografías nuevas; informe HTML generado por los grupos.
- **Sesión 2:** estructura, dataset, analizador de sentimientos, asistente conversacional y guía independiente incorporados al repositorio.
- **Evaluación posterior:** nuevos diseños visuales para estudiar la generalización del modelo, fuera del flujo principal de la sesión 2.
