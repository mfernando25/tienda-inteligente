# Conversational lab – Asistente de Tienda inteligente

Este laboratorio será el componente de integración de la Unidad 4. El asistente local consultará los resultados de visión por computadora y PLN, sin depender de APIs ni servicios externos.

## Fuentes de datos

| Fuente | Contenido |
|---|---|
| `cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json` | Conteos estimados por categoría y por anaquel, generados únicamente a partir de las predicciones visuales. |
| `nlp-lab/salida/resumen_sentimientos.json` | Comentarios y conteos de sentimientos agrupados por categoría. |

## Estructura prevista

```text
conversational-lab/
├── src/
│   └── (asistente conversacional)
└── README.md
```

El asistente identificará intenciones y entidades mediante reglas sencillas y responderá consultas sobre posiciones visibles en los anaqueles y opiniones de clientes.

**Nota:** los conteos del anaquel representan posiciones visibles (*facings*), no unidades almacenadas en bodega. El resumen de sentimientos contiene predicciones de un analizador léxico, no etiquetas verificadas de opiniones reales.
