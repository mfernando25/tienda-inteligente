# Sesión 2: La tienda entiende y conversa

**Unidad 4 – Inteligencia de Negocios**  
**Ejes 3 y 4:** Procesamiento de lenguaje natural y Sistemas conversacionales.

Esta guía continúa el proyecto iniciado en la Sesión 1. Reutiliza el inventario visual generado por el modelo HOG + SVM y añade el análisis de opiniones de clientes y un chatbot que aprende a reconocer intenciones.

## Objetivo

Construir una solución integrada que procese comentarios en español, clasifique sus sentimientos mediante un modelo preentrenado y permita consultar el inventario y las opiniones mediante preguntas en lenguaje natural.

## Preparación previa: entorno y modelo

**Esta preparación debe realizarse antes de iniciar los 60 minutos de práctica.** La instalación de PyTorch y la descarga inicial del modelo pueden tomar varios minutos, según la conexión y el equipo.

La Sesión 1 deja la terminal dentro de `cv-lab/`. Desde Git Bash, vuelve a la raíz del repositorio:

~~~bash
cd ..
git pull
~~~

Crea un entorno independiente para no modificar el entorno utilizado en visión por computadora:

~~~bash
python -m venv .venv-sesion2
source .venv-sesion2/Scripts/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r nlp-lab/requirements.txt
~~~

PyTorch se instala en su versión CPU para no requerir una GPU NVIDIA. La primera inicialización descarga el modelo de sentimientos en español; necesita conexión a Internet y debe realizarse antes de la sesión:

~~~bash
python -c "from pysentimiento import create_analyzer; create_analyzer(task='sentiment', lang='es')"
~~~

Se requiere Python 3.10 o superior y conexión a Internet para instalar las dependencias y descargar el modelo. Una vez preparado el entorno, abre la terminal desde la raíz del repositorio y verifica que el entorno `.venv-sesion2` esté activo.

**Verificar el resultado de la Sesión 1:** en el Explorador de VS Code, expande **cv-lab → salida_ml → iteraciones → evaluacion_01** y confirma que exista **inventario_por_foto.json**. No escribas la ruta del archivo en la terminal.

## 1. Comprender y explorar los comentarios

En el Explorador de VS Code, abre **nlp-lab → dataset → comentarios_clientes.csv**.

El dataset contiene 72 comentarios simulados en español sobre ocho categorías de productos. Cada categoría cuenta con tres etiquetas de referencia positivas, tres negativas y tres neutrales.

Observa el texto original y la columna `sentimiento_referencia`. Esta última es la etiqueta que permitirá evaluar las predicciones; no se entrega al modelo como parte del texto que debe clasificar.

## 2. Procesar y clasificar sentimientos

Desde la raíz del repositorio, ejecuta:

~~~bash
python nlp-lab/src/analizar_sentimientos.py
~~~

El programa combina dos operaciones diferentes:

1. **Normalización y tokenización:** transforma el texto a minúsculas, elimina tildes y utiliza NLTK para dividirlo en tokens. Esta salida es para explorar la estructura del comentario.
2. **Clasificación con PySentimiento:** envía el comentario original a un modelo Transformer preentrenado en español. El modelo devuelve una predicción positiva, negativa o neutral y las probabilidades asociadas.
3. **Evaluación:** compara las predicciones con `sentimiento_referencia` y calcula accuracy, F1 macro y matriz de confusión.
4. **Agrupación:** resume los resultados por producto para que el chatbot pueda consultarlos.

**Importante:** los tokens y el texto normalizado se incluyen para explorar el procesamiento textual; el modelo de sentimientos recibe el comentario original. Así se conserva el contexto y la puntuación del texto.

### Revisar los resultados

En el Explorador de VS Code, abre **nlp-lab → salida**. Examina:

- **comentarios_analizados.csv:** texto original, texto normalizado, tokens, sentimiento de referencia, predicción y probabilidades.
- **evaluacion_sentimientos.json:** accuracy, F1 macro y matriz de confusión.
- **resumen_sentimientos.json:** sentimientos y comentarios agrupados por producto. Este archivo alimenta el chatbot.

**Actividad: revisar predicciones incorrectas**

Trabaja con el archivo **`nlp-lab → salida → comentarios_analizados.csv`**, generado en el paso anterior. No uses `evaluacion_sentimientos.json` para esta actividad: ese archivo contiene métricas agregadas, no el detalle de cada comentario.

1. Abre `comentarios_analizados.csv` desde el Explorador de VS Code.
2. Busca el valor `False` (puedes usar Ctrl + F). Cada fila encontrada representa un comentario donde la predicción no coincide con la referencia.
3. En esa misma fila, lee las columnas `comentario`, `sentimiento_referencia` y `sentimiento_predicho`.
4. Selecciona dos casos y explica por qué crees que el modelo pudo clasificarlos de otra manera. Considera el contexto, expresiones propias de opiniones comerciales y las diferencias entre el lenguaje de redes sociales y las opiniones sobre productos.

Si no encuentras ninguna fila con `False`, significa que el modelo no tuvo desacuerdos con este conjunto de comentarios. En ese caso, revisa dos comentarios y explica qué elementos del texto pudieron influir en la predicción.

PySentimiento utiliza un modelo preentrenado para español, entrenado principalmente con publicaciones de redes sociales. Su resultado en comentarios comerciales puede variar. Además, las opiniones de esta práctica son simuladas; las métricas no representan el rendimiento en clientes reales.

## 3. Construir el sistema conversacional

El chatbot utilizará dos fuentes generadas por el proyecto:

- El inventario visual de **cv-lab → salida_ml → iteraciones → evaluacion_01 → inventario_por_foto.json**.
- El resumen de sentimientos de **nlp-lab → salida → resumen_sentimientos.json**.

Ejecuta desde la raíz del repositorio:

~~~bash
python conversational-lab/src/asistente_tienda.py
~~~

Al iniciar, el chatbot separa el 25% de las preguntas etiquetadas para una validación inicial, muestra accuracy y F1 macro, y luego entrena el clasificador final con todos los ejemplos de **conversational-lab → dataset → intenciones.csv**. Como el conjunto es pequeño, interpreta las métricas como una estimación inicial, no como garantía de que reconocerá cualquier pregunta.

### ¿Cómo reconoce las preguntas?

El sistema aplica aprendizaje automático supervisado:

1. **TF-IDF** transforma las preguntas de entrenamiento en características numéricas.
2. **Regresión Logística** aprende a asociar esas características con una intención, por ejemplo, consultar inventario, identificar productos con menor presencia o preguntar por opiniones.
3. El chatbot clasifica la nueva pregunta y muestra la intención detectada y el puntaje del clasificador.
4. Reconoce el producto mencionado mediante un conjunto de entidades y alias.
5. Consulta los JSON y construye una respuesta con los datos disponibles.

No es un modelo generativo como ChatGPT. Es un sistema conversacional acotado al negocio de la tienda, pero la identificación de intenciones ya no depende solamente de una lista de palabras clave: utiliza un clasificador entrenado con ejemplos etiquetados.

### Probar consultas de negocio

No necesitas recordar códigos de fotografías ni mencionar un anaquel concreto para las consultas globales.

**Inventario y exhibición:**

~~~text
¿Qué producto tiene menor presencia en los anaqueles?
¿Qué productos predominan en la exhibición?
¿Qué productos faltan en más anaqueles?
¿Qué anaquel está más lleno?
¿Cuánto espacio ocupa la leche?
~~~

**Opiniones de clientes:**

~~~text
¿Qué dicen los clientes sobre el cereal?
¿Qué quejas hay sobre la gaseosa?
¿Qué producto recibe más comentarios negativos?
~~~

**Consulta que requiere información histórica:**

~~~text
¿Desde hace cuánto falta el detergente?
~~~

El asistente debe indicar que no dispone de datos históricos. Las cinco fotografías corresponden a anaqueles distintos y no permiten calcular cuánto tiempo lleva faltando un producto.

### Actividad: ampliar el aprendizaje del chatbot

Abre **conversational-lab → dataset → intenciones.csv** en VS Code.

1. Revisa cómo cada pregunta está asociada a una etiqueta de intención.
2. Añade dos formas distintas de expresar una de las consultas, conservando la intención correcta.
3. Guarda el archivo y reinicia el chatbot para que vuelva a entrenarse con los nuevos ejemplos.
4. Prueba una pregunta nueva y observa qué intención detecta y qué puntaje devuelve.

Esta actividad permite comprobar que el clasificador aprende de los ejemplos etiquetados. Tras añadir preguntas, las métricas de validación pueden variar. Si una pregunta está muy alejada de las expresiones de entrenamiento, puede clasificarla incorrectamente; por eso se deben ampliar y evaluar sus ejemplos.

## 4. Distribución de la práctica (60 minutos)

| Tiempo | Actividad |
|---|---|
| 0–5 min | Conectar los resultados de visión artificial con la nueva funcionalidad de la tienda. |
| 5–15 min | Explorar normalización y tokenización de comentarios. |
| 15–27 min | Ejecutar el análisis de sentimientos y revisar desacuerdos. |
| 27–45 min | Evaluar, ampliar y volver a entrenar el clasificador de intenciones. |
| 45–55 min | Consultar inventario y opiniones desde el chatbot. |
| 55–60 min | Interpretar resultados y discutir limitaciones. |

La instalación de dependencias y la primera descarga del modelo se realizan antes de estos 60 minutos.

## 5. Entregables

1. Los tres archivos generados en **nlp-lab → salida**:
   - `comentarios_analizados.csv`
   - `evaluacion_sentimientos.json`
   - `resumen_sentimientos.json`
2. Dos nuevas preguntas etiquetadas agregadas al dataset de intenciones.
3. Registro de al menos tres consultas al chatbot: una sobre inventario, una sobre opiniones y otra sobre comentarios negativos.
4. Una explicación breve de un desacuerdo del análisis de sentimientos y una limitación del clasificador de intenciones.

## 6. Consideraciones

- El inventario representa posiciones visibles (*facings*) en cinco fotografías, no unidades disponibles en almacén.
- La ausencia de un producto en una fotografía no demuestra que esté agotado en el almacén.
- No se puede inferir la duración de un faltante sin registros históricos.
- El modelo de sentimientos está preentrenado en español, pero su dominio de entrenamiento no coincide necesariamente con opiniones comerciales.
- El chatbot utiliza aprendizaje automático para clasificar intenciones; las entidades y las respuestas se resuelven mediante reglas y plantillas. No es un LLM ni comprende consultas arbitrarias.
- El entorno `.venv-sesion2` y los resultados de `nlp-lab/salida/` son locales y no se suben al repositorio.
