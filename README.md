# Tienda inteligente

Proyecto académico de la asignatura **Inteligencia de Negocios**.

El proyecto integra Machine Learning, visión por computadora, procesamiento de lenguaje natural y sistemas conversacionales a través de un caso de supermercado.

## Componentes

### Generador de anaqueles
Aplicación web local 2D con un anaquel de 4 filas × 8 columnas, ocho productos ilustrados, escenarios controlados, edición manual y exportaciones PNG, JSON y CSV.

Instrucciones: [shelf-generator/README.md](shelf-generator/README.md).

### Visión por computadora y Machine Learning
Incluye rectificación de perspectiva, preparación del dataset, entrenamiento HOG + SVM e inferencia por lotes sobre fotografías completas.

Instrucciones técnicas: [cv-lab/README.md](cv-lab/README.md).

### Guía de laboratorio – Sesión 1
[Cierre de sesión 1: inferencia del anaquel completo](labs/GUIA_PRACTICA_INTEGRACION_ML_VISION_ANAQUEL.md).

### Pautas de continuidad de la Unidad 4
[Decisiones del proyecto, iteraciones y escenarios del examen](docs/PAUTAS_CONTINUIDAD_UNIDAD4.md).

## Estructura

- shelf-generator/: generador de anaqueles.
- cv-lab/: scripts OpenCV, preparación del dataset, entrenamiento e inferencia.
- docs/: arquitectura y decisiones del proyecto.
- labs/: guías de prácticas.
- exam/: recursos de evaluación.

## Requisitos y restricciones

- Herramientas gratuitas y preferentemente open source.
- Ejecución local siempre que sea posible.
- Sin dependencia de servicios externos de pago.
- Instrucciones reproducibles para Windows.

## Estado

- **Generador:** prototipo funcional.
- **Dataset de entrenamiento:** 15 fotografías y hasta 480 recortes por grupo; datos locales.
- **Modelo inicial:** HOG + SVM entrenado con ocho ilustraciones de productos y la clase `empty`.
- **Cierre de sesión 1:** inferencia por lotes sobre 15 anaqueles nuevos, pendiente de ejecución por los grupos.
- **Iteración 02:** nuevos diseños visuales para observar la generalización; posterior a la Iteración 01.
- **Sesión 2:** PLN y Sistemas conversacionales; tendrá una guía independiente.
