# Conversational lab – Asistente de Tienda inteligente

Este laboratorio integra los resultados de visión por computadora y PLN mediante un asistente local basado en reglas. No depende de APIs, LLM externos ni backend.

## Fuentes de datos

| Fuente | Contenido |
|---|---|
| `cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json` | Conteos estimados por categoría y por anaquel, generados únicamente a partir de las predicciones visuales. |
| `nlp-lab/salida/resumen_sentimientos.json` | Comentarios y conteos de sentimientos agrupados por categoría. |

## Ejecutar

Desde la raíz del repositorio:

```bash
python conversational-lab/src/asistente_tienda.py
```

El asistente permite consultar:
- Posiciones visibles estimadas de una categoría en un anaquel identificado.
- Categorías con menos de un número de posiciones indicado.
- Resumen de sentimientos por categoría.
- Ejemplos de comentarios clasificados como negativos.
- Categorías con más opiniones negativas clasificadas.

Escribe `ayuda` para ver ejemplos y `salir` para terminar.

## Ejemplos

```text
¿Cuántas posiciones de leche hay en captura_031?
¿Qué productos tienen menos de 3 posiciones en captura_031?
¿Qué opinan los clientes sobre el cereal?
¿Qué comentarios negativos hay sobre la gaseosa?
¿Qué categoría tiene más opiniones negativas?
```

## Limitaciones

La identificación de intenciones y entidades utiliza palabras clave y expresiones regulares; no es comprensión de lenguaje natural abierta. Los conteos representan posiciones visibles (*facings*), no stock de almacén. Los sentimientos son predicciones de un analizador léxico, no etiquetas verificadas de opiniones reales.
