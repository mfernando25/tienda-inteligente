# Conversational lab – Asistente de Tienda inteligente

Este laboratorio conecta los resultados de visión artificial y PLN mediante un chatbot que **aprende a clasificar intenciones con aprendizaje automático**. No utiliza un LLM ni genera respuestas libres.

## Fuentes de información

| Fuente | Uso |
|---|---|
| `cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json` | Posiciones estimadas por categoría en las cinco fotografías de anaqueles. |
| `nlp-lab/salida/resumen_sentimientos.json` | Sentimientos predichos y comentarios agrupados por categoría. |
| `conversational-lab/dataset/intenciones.csv` | Preguntas etiquetadas que se utilizan para entrenar el clasificador de intenciones. |

## Cómo funciona

1. `TfidfVectorizer` transforma las preguntas de entrenamiento en características numéricas.
2. `LogisticRegression` aprende a asociar esas características con una intención.
3. El programa reserva el 25% de las preguntas para una validación inicial, muestra accuracy y F1 macro y luego ajusta el chatbot final con todos los ejemplos.
4. El chatbot clasifica la pregunta nueva y muestra la intención detectada y el puntaje del clasificador.
5. Identifica el producto mencionado mediante un diccionario de entidades y alias.
6. Consulta los JSON del proyecto y construye la respuesta con los datos disponibles.

La validación usa un conjunto pequeño de ejemplos, por lo que sus métricas son orientativas. El clasificador se entrena cada vez que se inicia el asistente. Para ampliar las formas de preguntar, se agregan nuevas frases etiquetadas en `dataset/intenciones.csv` y se reinicia el programa.

## Ejecutar

Desde la raíz del repositorio y con el entorno `.venv-sesion2` activado:

~~~bash
python conversational-lab/src/asistente_tienda.py
~~~

## Consultas de negocio

~~~text
¿Qué producto tiene menor presencia en los anaqueles?
¿Qué productos predominan en la exhibición?
¿Qué productos faltan en más anaqueles?
¿Qué anaquel está más lleno?
¿Cuánto espacio ocupa la leche?
¿Qué dicen los clientes sobre el cereal?
¿Qué quejas hay sobre la gaseosa?
¿Qué producto recibe más comentarios negativos?
¿Desde hace cuánto falta el detergente?
~~~

## Alcance y límites

Es un sistema conversacional acotado al dominio de la tienda. Su clasificación de intenciones sí utiliza un modelo supervisado de aprendizaje automático (TF-IDF + Regresión Logística), pero las entidades de producto se reconocen mediante alias y las respuestas se construyen con plantillas. No es un modelo generativo ni comprende cualquier consulta.

Los cinco anaqueles son fotografías distintas, no registros históricos del mismo anaquel. El chatbot puede comparar presencia y posiciones visibles, pero no sabe cuánto tiempo lleva faltando un producto ni puede inferir el stock de almacén. Los sentimientos provienen del modelo preentrenado de PLN y deben interpretarse considerando que los comentarios del ejercicio son simulados.
