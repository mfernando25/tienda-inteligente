# NLP lab – Opiniones de clientes

Laboratorio de PLN de la Unidad 4. Procesa comentarios de clientes, muestra su normalización y tokenización con NLTK y clasifica el sentimiento mediante un modelo Transformer preentrenado en español (PySentimiento).

## Estructura

~~~text
nlp-lab/
├── dataset/
│   └── comentarios_clientes.csv
├── requirements.txt
├── src/
│   └── analizar_sentimientos.py
└── salida/
    ├── comentarios_analizados.csv
    ├── evaluacion_sentimientos.json
    └── resumen_sentimientos.json
~~~

## Dataset

`dataset/comentarios_clientes.csv` contiene 72 comentarios simulados en español: ocho categorías y nueve comentarios por categoría. Cada categoría incluye tres etiquetas de referencia positivas, tres negativas y tres neutrales.

La columna `sentimiento_referencia` se reserva para evaluar la predicción. El modelo recibe el texto original del comentario; no utiliza la etiqueta de referencia como entrada.

## Preparación del entorno

Utiliza el entorno independiente `.venv-sesion2` desde la raíz del repositorio. Instala las dependencias una sola vez:

~~~bash
python -m pip install -r nlp-lab/requirements.txt
~~~

La primera inicialización de PySentimiento descarga el modelo preentrenado en español. Conviene realizar la instalación y esa primera descarga antes de la sesión de 60 minutos.

## Ejecutar desde la raíz

~~~bash
python nlp-lab/src/analizar_sentimientos.py
~~~

El programa:
1. Normaliza el texto y muestra sus tokens mediante NLTK.
2. Clasifica el comentario original con un Transformer preentrenado en español.
3. Registra la clase predicha y sus probabilidades.
4. Compara la predicción con `sentimiento_referencia`.
5. Calcula accuracy, F1 macro y matriz de confusión.
6. Genera un resumen de sentimientos y comentarios por producto para el chatbot.

## Archivos generados

- `salida/comentarios_analizados.csv`: comentario original, texto normalizado, tokens, etiqueta de referencia, predicción y probabilidades.
- `salida/evaluacion_sentimientos.json`: métricas de evaluación y matriz de confusión.
- `salida/resumen_sentimientos.json`: conteos y comentarios por producto. Es la entrada de opiniones del chatbot.

## Alcance y limitaciones

PySentimiento utiliza un modelo Transformer preentrenado para análisis de sentimientos en español. El modelo de referencia se entrenó principalmente con publicaciones de redes sociales, por lo que sus resultados pueden variar al aplicarlo a opiniones comerciales. Los 72 comentarios usados aquí son sintéticos: la comparación mide el comportamiento del modelo en este ejercicio, no su desempeño en clientes reales.

Referencias: [PySentimiento](https://github.com/pysentimiento/pysentimiento) · [Modelo de sentimientos en español](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis).
