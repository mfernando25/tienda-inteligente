# Sesión 2: La tienda entiende y conversa

**Unidad 4 – Inteligencia de Negocios**  
**Ejes 3 y 4:** Procesamiento de lenguaje natural y Sistemas conversacionales.

Esta es la segunda guía de la Unidad 4. Complementa la guía de visión por computadora y reutiliza el inventario estimado que produjo la Sesión 1. No modifica ni reemplaza la primera guía.

## Objetivo

Procesar comentarios de clientes en español, clasificarlos por sentimiento y construir un asistente local que responda consultas usando los resultados de PLN y visión por computadora.

## 1. Preparar el entorno

Esta segunda guía comienza **después de completar la guía de Sesión 1**. No repetir el entrenamiento ni la evaluación visual.

Al terminar la primera guía, la terminal suele quedar dentro de `cv-lab/`. Volver a la raíz del repositorio:

~~~bash
cd ..
~~~

Actualizar el repositorio si hay cambios nuevos y comprobar Python:

~~~bash
git pull
python --version
~~~

Se requiere Python 3.10 o superior. Esta práctica utiliza la biblioteca estándar; no requiere instalar paquetes externos ni activar el entorno virtual de OpenCV.

**Comprobar el archivo generado en la Sesión 1:** en el panel **Explorador** de VS Code, expande las carpetas **cv-lab → salida_ml → iteraciones → evaluacion_01** y verifica que aparezca **inventario_por_foto.json**.

No escribas ni ejecutes la ruta del archivo en la terminal. Solo ubícalo y verifica su presencia en el Explorador. Si no aparece, primero se debe completar la evaluación de visión por computadora de la primera guía.

## 2. Explorar el dataset de opiniones

En el panel **Explorador** de VS Code, expande **nlp-lab → dataset** y abre el archivo **comentarios_clientes.csv**. No escribas su ruta en la terminal.

El dataset contiene 72 comentarios simulados en español, distribuidos en ocho categorías:

- Conservas
- Cereal
- Galletas
- Detergente
- Jugo
- Leche
- Gaseosa
- Agua

Cada categoría tiene nueve comentarios: tres positivos, tres negativos y tres neutrales.

Columnas principales:
- `comentario_id`: identificador del comentario.
- `product_id`: identificador de categoría.
- `producto`: nombre en español.
- `comentario`: texto del cliente.
- `sentimiento_referencia`: etiqueta de referencia para evaluar la clasificación.

La etiqueta de referencia no se usa para predecir el sentimiento; se consulta después para comprobar el resultado.

## 3. Analizar los sentimientos

Desde la raíz del repositorio, ejecutar:

~~~bash
python nlp-lab/src/analizar_sentimientos.py
~~~

El programa realiza automáticamente:

1. **Normalización:** pasa el texto a minúsculas y elimina tildes para facilitar la comparación.
2. **Tokenización:** separa el comentario en palabras.
3. **Análisis léxico:** cuenta términos incluidos en diccionarios positivos y negativos.
4. **Negación:** invierte la polaridad de términos cercanos a palabras como «no», «nunca» y «tampoco».
5. **Clasificación:** asigna positivo, negativo o neutral.
6. **Evaluación:** compara la predicción con `sentimiento_referencia`.
7. **Agregación:** calcula los resultados por categoría y guarda los comentarios clasificados.

### Archivos generados

Los resultados se crean automáticamente en la carpeta **nlp-lab → salida**. Desde el Explorador de VS Code, expande esas carpetas para consultar:

- **comentarios_analizados.csv:** incluye el texto limpio, tokens, puntajes, sentimiento predicho y etiqueta de referencia.
- **evaluacion_sentimientos.json:** incluye accuracy, F1 macro, matriz de confusión y métricas por sentimiento.
- **resumen_sentimientos.json:** resume los sentimientos y conserva los comentarios agrupados por producto. Este archivo alimentará el asistente conversacional.

Abre **comentarios_analizados.csv** desde el Explorador de VS Code para observar los comentarios que el método clasificó de forma diferente a la referencia. No copies la ruta del archivo en la terminal. Revisa especialmente los casos de negación y expresiones implícitas.

**Importante:** el método es léxico y basado en reglas. No comprende completamente el contexto, la ironía ni todas las expresiones. Un resultado alto sobre este dataset simulado no equivale a un sistema de análisis de opiniones listo para producción.

## 4. Ejecutar e interactuar con el asistente conversacional

El asistente consulta automáticamente dos resultados generados en las sesiones anteriores:

- El inventario visual **inventario_por_foto.json**, ubicado en **cv-lab → salida_ml → iteraciones → evaluacion_01**.
- El resumen de opiniones **resumen_sentimientos.json**, ubicado en **nlp-lab → salida**.

No necesitas abrir, copiar ni ejecutar estos archivos. El programa los lee directamente.

Ejecutar desde la raíz del repositorio:

~~~bash
python conversational-lab/src/asistente_tienda.py
~~~

El programa funciona en la terminal. Escribe tus consultas después del indicador `Tú:`.

### ¿Es una IA o un chatbot?

Es un **chatbot local basado en reglas** y constituye un sistema conversacional sencillo. Recibe preguntas escritas en lenguaje natural, identifica palabras clave y patrones para reconocer la intención y el producto, consulta los resultados de visión y PLN, y construye una respuesta.

No es un modelo generativo como ChatGPT: no comprende cualquier pregunta, no mantiene una conversación abierta ni genera respuestas a partir de conocimiento general. Puede responder consultas del dominio de la tienda siempre que estén formuladas con términos que reconoce.

### Cómo formular las preguntas

No es necesario copiar exactamente los ejemplos, pero la consulta debe expresar una de las tareas que el asistente reconoce:

- **Inventario:** pregunta por el inventario, stock, disponibilidad, posiciones o productos exhibidos. Menciona la categoría si buscas un producto concreto.
- **Anaquel:** puedes identificarlo como `anaquel 031` o `captura_031`. Si no indicas uno, el asistente puede mostrar los resultados de todos los anaqueles.
- **Opiniones:** pregunta qué dicen, qué opinan o qué comentarios tienen los clientes. Incluye el nombre del producto.
- **Problemas o quejas:** menciona términos como «comentarios negativos», «quejas», «reclamos» o «problemas» para solicitar los ejemplos negativos clasificados.

Formula una consulta por vez. Puedes escribirla con tus propias palabras, pero debe contener las pistas necesarias para que el asistente identifique qué dato necesitas.

### Consultas de ejemplo

**Inventario visual:**

~~~text
¿Cómo está el inventario del anaquel 031?
¿Cuánto cereal se exhibe en el anaquel 031?
¿Hay leche disponible en el anaquel 031?
~~~

**Opiniones de clientes:**

~~~text
¿Qué dicen los clientes sobre el cereal?
¿Hay comentarios negativos sobre la gaseosa?
¿Qué problemas reportan los clientes sobre el detergente?
~~~

El asistente responde con los datos disponibles en los JSON. El inventario representa posiciones visibles (*facings*), no unidades almacenadas en bodega. Los sentimientos son resultados del analizador léxico, no opiniones reales verificadas.

Escribir `ayuda` para ver ejemplos o `salir` para terminar. Si la consulta no coincide con las reglas implementadas, el asistente indicará que no pudo identificarla y mostrará ejemplos.

## 5. Interpretar y discutir

Responder:

1. ¿Qué palabras o elementos de la pregunta ayudan al asistente a identificar la intención?
2. ¿Qué sucede si se pregunta por un producto que no está incluido en las categorías?
3. ¿Por qué dos preguntas con el mismo propósito pueden producir resultados diferentes si una no contiene las palabras que el asistente reconoce?
4. ¿Qué diferencia existe entre un chatbot basado en reglas y un asistente generativo como ChatGPT?
5. ¿Qué cambios permitirían ampliar las consultas que puede reconocer el asistente?

## Entregables

En el Explorador de VS Code, ubica la carpeta **nlp-lab → salida**. Allí se encuentran los tres resultados que se generan al ejecutar el analizador:

- **comentarios_analizados.csv**
- **evaluacion_sentimientos.json**
- **resumen_sentimientos.json**

Además, realiza al menos tres consultas al asistente: una sobre inventario, una sobre opiniones y una sobre comentarios negativos. Registra las preguntas realizadas y las respuestas obtenidas. Redacta una conclusión breve sobre las limitaciones del análisis léxico y del sistema conversacional.

Los datos y resultados generados localmente no se suben al repositorio. El código y el dataset base de la práctica sí se versionan.
