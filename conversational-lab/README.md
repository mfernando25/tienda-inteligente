# Conversational lab – Asistente de Tienda inteligente

Este laboratorio integra los resultados de visión por computadora y PLN mediante un **chatbot local basado en reglas**. No depende de APIs, LLM externos ni backend.

## Fuentes de datos

| Fuente | Contenido |
|---|---|
| `cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json` | Conteos estimados de productos por anaquel, derivados de las predicciones visuales. |
| `nlp-lab/salida/resumen_sentimientos.json` | Comentarios y conteos de sentimientos agrupados por categoría. |

## Ejecutar

Desde la raíz del repositorio:

~~~bash
python conversational-lab/src/asistente_tienda.py
~~~

## Consultas orientadas al negocio

El asistente analiza por defecto el conjunto de cinco fotografías. No hace falta indicar el código de una captura.

**Presencia y exhibición:**

~~~text
¿Qué producto tiene menor presencia en los anaqueles?
¿Qué productos predominan en la exhibición?
¿Qué productos faltan en más anaqueles?
¿Qué anaqueles están más llenos?
¿Cuánto espacio ocupa la leche en los anaqueles?
~~~

**Opiniones de clientes:**

~~~text
¿Qué dicen los clientes sobre el cereal?
¿Qué producto recibe más comentarios negativos?
¿Qué quejas hay sobre la gaseosa?
~~~

Se aceptan preguntas redactadas con palabras propias, siempre que incluyan términos relacionados con la intención. El código de captura solo se necesita para consultar una fotografía específica.

## Qué tipo de sistema es

Es un chatbot conversacional local basado en reglas. Identifica palabras clave y patrones, consulta los JSON y construye una respuesta. Acepta lenguaje natural dentro del dominio acotado de la tienda, pero no es un modelo generativo ni comprende consultas arbitrarias.

## Límites de los datos

- Las cinco fotografías representan cinco anaqueles distintos, no mediciones de un mismo anaquel a lo largo del tiempo.
- Los totales agregados comparan posiciones visibles (*facings*) en esas cinco fotografías. No representan unidades en almacén ni el stock total de una tienda.
- La ausencia de un producto en una fotografía no demuestra que esté agotado en almacén.
- No se puede determinar cuánto tiempo lleva faltando un producto: no hay fechas históricas comparables ni registros de reposición.
- Los sentimientos son predicciones de un analizador léxico sobre comentarios simulados, no opiniones reales verificadas.
