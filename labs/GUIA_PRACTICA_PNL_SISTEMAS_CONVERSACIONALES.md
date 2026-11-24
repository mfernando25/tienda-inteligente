# Sesión 2: La tienda entiende y conversa

**Unidad 4 – Inteligencia de Negocios**  
**Ejes 3 y 4:** Procesamiento de lenguaje natural y Sistemas conversacionales.

Esta es la segunda guía de la Unidad 4. Complementa la guía de visión por computadora y reutiliza el inventario estimado que produjo la Sesión 1. No modifica ni reemplaza la primera guía.

## Objetivo

Procesar comentarios de clientes en español, clasificarlos por sentimiento y construir un asistente local que responda consultas usando los resultados de PLN y visión por computadora.

## 1. Preparar el entorno

Desde Git Bash, ubicarse en la carpeta raíz del repositorio:

~~~bash
git pull
python --version
~~~

Esta práctica utiliza la biblioteca estándar de Python; no requiere instalar paquetes externos ni activar el entorno virtual de OpenCV.

Comprobar que existe el inventario generado en la sesión anterior:

~~~text
cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json
~~~

Si el archivo no existe, primero se debe completar la evaluación de visión por computadora de la primera guía.

## 2. Explorar el dataset de opiniones

Abrir:

~~~text
nlp-lab/dataset/comentarios_clientes.csv
~~~

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

Se crean automáticamente en `nlp-lab/salida/`:

~~~text
salida/
├── comentarios_analizados.csv
├── evaluacion_sentimientos.json
└── resumen_sentimientos.json
~~~

- `comentarios_analizados.csv`: incluye el texto limpio, tokens, puntajes, sentimiento predicho y etiqueta de referencia.
- `evaluacion_sentimientos.json`: incluye accuracy, F1 macro, matriz de confusión y métricas por sentimiento.
- `resumen_sentimientos.json`: resume los sentimientos y conserva los comentarios agrupados por producto. Este archivo alimentará el asistente conversacional.

Abrir `comentarios_analizados.csv` para observar los comentarios que el método clasificó de forma diferente a la referencia. Revisar especialmente los casos de negación y expresiones implícitas.

**Importante:** el método es léxico y basado en reglas. No comprende completamente el contexto, la ironía ni todas las expresiones. Un resultado alto sobre este dataset simulado no equivale a un sistema de análisis de opiniones listo para producción.

## 4. Ejecutar el sistema conversacional

El asistente consulta dos archivos:

~~~text
cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json
nlp-lab/salida/resumen_sentimientos.json
~~~

Ejecutar desde la raíz del repositorio:

~~~bash
python conversational-lab/src/asistente_tienda.py
~~~

El programa funciona en la terminal. Identifica intenciones y categorías mediante reglas sencillas; no utiliza un LLM ni una API externa.

### Consultas para probar

**Inventario visual:**

~~~text
¿Cuántas posiciones de leche hay en captura_031?
¿Qué productos tienen menos de 3 posiciones en captura_031?
¿Cuál es el inventario de captura_032?
~~~

**Opiniones de clientes:**

~~~text
¿Qué opinan los clientes sobre el cereal?
¿Qué comentarios negativos hay sobre la gaseosa?
¿Qué categoría tiene más opiniones negativas?
~~~

El asistente responde con los datos disponibles en los JSON. El inventario representa posiciones visibles (*facings*), no unidades almacenadas en bodega. Los sentimientos son resultados del analizador léxico, no opiniones reales verificadas.

Escribir `ayuda` para ver ejemplos o `salir` para terminar.

## 5. Interpretar y discutir

Responder:

1. ¿Cómo se transforma un comentario en tokens?
2. ¿Por qué una negación puede cambiar el sentimiento de una frase?
3. ¿Qué expresiones no logra interpretar correctamente el diccionario?
4. ¿Qué diferencia existe entre consultar inventario visual y consultar opiniones?
5. ¿Qué limitaciones tiene un asistente que identifica intenciones mediante reglas?

## Entregables

- `nlp-lab/salida/comentarios_analizados.csv`.
- `nlp-lab/salida/evaluacion_sentimientos.json`.
- `nlp-lab/salida/resumen_sentimientos.json`.
- Una prueba del asistente con consultas de inventario y opiniones.
- Una conclusión breve sobre las limitaciones del análisis léxico y del sistema conversacional.

Los datos y resultados generados localmente no se suben al repositorio. El código y el dataset base de la práctica sí se versionan.
