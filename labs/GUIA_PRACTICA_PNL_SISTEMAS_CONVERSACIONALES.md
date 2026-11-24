# Sesión 2: La tienda entiende y conversa

**Unidad 4 – Inteligencia de Negocios**  
**Ejes 3 y 4:** Procesamiento de lenguaje natural y Sistemas conversacionales.

Esta guía continúa el proyecto iniciado en la Sesión 1. Reutiliza el inventario visual generado por el modelo HOG + SVM y añade el análisis de opiniones de clientes y un chatbot que aprende a reconocer intenciones.

## Objetivo

Construir una solución integrada que procese comentarios en español, clasifique sus sentimientos mediante un modelo preentrenado y permita consultar el inventario y las opiniones mediante preguntas en lenguaje natural.

## Preparación previa: entorno y modelo

**Realiza esta preparación antes de la sesión.** La instalación de PyTorch y la descarga inicial del modelo pueden tomar varios minutos, según la conexión y el equipo.

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
4. Selecciona un caso y explica por qué crees que el modelo pudo clasificarlo de otra manera. Considera el contexto, expresiones propias de opiniones comerciales y las diferencias entre el lenguaje de redes sociales y las opiniones sobre productos.

Si no encuentras ninguna fila con `False`, significa que el modelo no tuvo desacuerdos con este conjunto de comentarios. En ese caso, revisa un comentario y explica qué elementos del texto pudieron influir en la predicción.

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

Una **intención** es el propósito de una pregunta: qué información busca el usuario o qué acción espera del chatbot. No es el nombre del producto ni una palabra aislada. Por eso, dos preguntas sobre la leche pueden tener intenciones distintas:

| Pregunta | Intención |
|---|---|
| ¿Cuántas posiciones visibles tiene la leche? | `INVENTARIO_PRODUCTO` |
| ¿Qué opinan los clientes sobre la leche? | `OPINIONES_PRODUCTO` |
| ¿Qué quejas hay sobre la leche? | `QUEJAS_PRODUCTO` |

El sistema aplica aprendizaje automático supervisado:

1. **Stemming en español** reduce variantes de una misma palabra (por ejemplo, «producto/productos» y «predomina/predominan»). **TF-IDF** combina características de palabras (con stemming) y secuencias de caracteres para reconocer mejor variantes escritas.
2. **Regresión Logística** aprende a asociar esas características con una intención, por ejemplo, consultar inventario, identificar productos con menor presencia o preguntar por opiniones.
3. El chatbot clasifica la nueva pregunta y muestra la intención detectada y el puntaje del clasificador. Si la puntuación es baja o las dos intenciones más probables son demasiado cercanas, solicita reformular la pregunta en lugar de ejecutar una intención dudosa.
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

### Actividad: realizar siete consultas

Realiza **siete consultas diferentes**, cada una correspondiente a una intención distinta. Incluye obligatoriamente:
- Una consulta sobre inventario de un producto (`INVENTARIO_PRODUCTO`).
- Una consulta sobre opiniones de un producto (`OPINIONES_PRODUCTO`).
- Una consulta sobre comentarios negativos de un producto (`QUEJAS_PRODUCTO`).

Selecciona otras cuatro intenciones entre las disponibles. Para cada consulta, registra la pregunta realizada, la intención detectada y un resumen de la respuesta del chatbot.

### Actividad: ampliar el aprendizaje del chatbot

Abre **conversational-lab → dataset → intenciones.csv** en VS Code.

El archivo contiene inicialmente **72 preguntas etiquetadas: ocho por cada una de las nueve intenciones**.

**¿Qué debes agregar?**

1. Añade **15 preguntas nuevas en total**. Incluye al menos una pregunta para cada una de las nueve intenciones y distribuye las seis restantes entre las intenciones que consideres que necesitan más variedad.
2. Cada fila debe tener dos datos: la etiqueta de intención existente y una pregunta nueva. Utiliza exactamente los nombres de las intenciones que ya aparecen en el CSV; no inventes etiquetas.
3. Escribe las preguntas con tus propias palabras. No basta con copiar o cambiar una sola palabra de los ejemplos existentes. Evita repetir preguntas.
4. Guarda el archivo y reinicia el chatbot para que vuelva a entrenarse con los nuevos ejemplos.
5. Comprueba algunas de tus preguntas nuevas y observa qué intención detecta y qué puntaje devuelve.

Recuerda: la intención representa lo que el usuario quiere saber o hacer. Por ejemplo, preguntar por la cantidad visible de leche es distinto de preguntar qué opinan los clientes sobre ella. Aunque el producto sea el mismo, la intención cambia.

Esta actividad permite comprobar que el clasificador aprende de los ejemplos etiquetados. Tras añadir preguntas, las métricas de validación pueden variar. Si una pregunta está muy alejada de las expresiones de entrenamiento, puede clasificarla incorrectamente; por eso se deben ampliar y evaluar sus ejemplos.

## 4. Entregables

1. Los tres archivos generados en **nlp-lab → salida**:
   - `comentarios_analizados.csv`
   - `evaluacion_sentimientos.json`
   - `resumen_sentimientos.json`
2. El archivo `intenciones.csv` actualizado con **15 preguntas nuevas y etiquetadas**.
3. Registro de **siete consultas diferentes** al chatbot. Cada una debe corresponder a una intención distinta e incluir la pregunta, la intención detectada y un resumen de la respuesta.
4. Una explicación breve de un desacuerdo del análisis de sentimientos y una limitación del clasificador de intenciones.

**Ejemplo de desacuerdo de sentimientos:** el comentario «La fecha de vencimiento aparece impresa en la parte superior» tiene como referencia `neutral`. Si el modelo lo clasificara como `negativo`, una posible explicación sería que asoció la palabra «vencimiento» con una señal negativa, aunque el comentario solo comunica un dato del envase. Es un ejemplo orientativo: en tu entrega debes revisar los valores reales de tu CSV y explicar un desacuerdo que aparezca en tus resultados.

## 5. Consideraciones

- El inventario representa posiciones visibles (*facings*) en cinco fotografías, no unidades disponibles en almacén.
- La ausencia de un producto en una fotografía no demuestra que esté agotado en el almacén.
- No se puede inferir la duración de un faltante sin registros históricos.
- El modelo de sentimientos está preentrenado en español, pero su dominio de entrenamiento no coincide necesariamente con opiniones comerciales.
- El chatbot utiliza aprendizaje automático para clasificar intenciones; las entidades y las respuestas se resuelven mediante reglas y plantillas. No es un LLM ni comprende consultas arbitrarias.
- El entorno `.venv-sesion2` y los resultados de `nlp-lab/salida/` son locales y no se suben al repositorio.
