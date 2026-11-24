# Dataset de opiniones de clientes

Archivo: `comentarios_clientes.csv`

## Composición

- 72 comentarios simulados en español peruano neutral.
- Ocho categorías: Conservas, Cereal, Galletas, Detergente, Jugo, Leche, Gaseosa y Agua.
- Nueve comentarios por categoría.
- Por cada categoría: tres positivos, tres negativos y tres neutrales.
- Incluye ejemplos con negación y algunas expresiones implícitas para observar las limitaciones de un analizador léxico.

## Columnas

| Columna | Propósito |
|---|---|
| `comentario_id` | Identificador único del comentario. |
| `product_id` | ID de categoría compatible con el generador del anaquel. |
| `producto` | Nombre del producto en español. |
| `comentario` | Opinión escrita por el cliente. |
| `sentimiento_referencia` | Etiqueta de referencia: positivo, negativo o neutral. |

Los comentarios son sintéticos, creados para la práctica. No representan testimonios ni reseñas de clientes reales.

La etiqueta de referencia se usa únicamente para evaluar la clasificación después de predecir el sentimiento.
