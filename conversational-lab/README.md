# Conversational lab – Asistente de Tienda inteligente

Este laboratorio integra los resultados de visión por computadora y PLN mediante un **chatbot local basado en reglas**. No depende de APIs, LLM externos ni backend.

## Fuentes de datos

| Fuente | Contenido |
|---|---|
| `cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json` | Conteos estimados por categoría y por anaquel, generados únicamente a partir de las predicciones visuales. |
| `nlp-lab/salida/resumen_sentimientos.json` | Comentarios y conteos de sentimientos agrupados por categoría. |

## Ejecutar

Desde la raíz del repositorio:

~~~bash
python conversational-lab/src/asistente_tienda.py
~~~

El asistente permite consultar posiciones visibles de productos, inventario por anaquel, resumen de sentimientos y comentarios clasificados como negativos.

## Ejemplos de consultas

**Inventario:**

~~~text
¿Cómo está el inventario del anaquel 031?
¿Cuánto cereal se exhibe en el anaquel 031?
¿Hay leche disponible en el anaquel 031?
~~~

**Opiniones:**

~~~text
¿Qué dicen los clientes sobre el cereal?
¿Hay comentarios negativos sobre la gaseosa?
¿Qué problemas reportan los clientes sobre el detergente?
~~~

## Cómo formular consultas

Se puede escribir en español sin copiar literalmente los ejemplos, pero las preguntas deben incluir términos que permitan reconocer la intención:

- Para inventario: términos como inventario, stock, existencias, disponibilidad, posiciones o exhibición.
- Para un anaquel: indicar su número, por ejemplo `anaquel 031` o `captura_031`.
- Para opiniones: mencionar clientes, opiniones, comentarios, reseñas o qué dicen/opinan.
- Para comentarios negativos: mencionar negativos, quejas, reclamos o problemas.
- Para consultar un producto: incluir su nombre o categoría.

El asistente trabaja con estas intenciones y categorías concretas. Si la frase no contiene las pistas reconocidas, no podrá identificar la consulta.

## Qué tipo de sistema es

Es un chatbot conversacional sencillo basado en reglas: identifica palabras clave y patrones, consulta los JSON y arma una respuesta. Acepta preguntas en lenguaje natural acotado al dominio de la tienda, pero **no entiende cualquier pregunta ni es un modelo generativo como ChatGPT**. No aprende de la conversación ni razona sobre información que no esté en sus fuentes.

## Limitaciones

La identificación de intenciones y entidades utiliza palabras clave y expresiones regulares; no es comprensión de lenguaje natural abierta. Los conteos representan posiciones visibles (*facings*), no stock de almacén. Los sentimientos son predicciones de un analizador léxico, no etiquetas verificadas de opiniones reales.
