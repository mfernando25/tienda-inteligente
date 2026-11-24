# NLP lab – Opiniones de clientes

Laboratorio local de PLN para la Unidad 4 de Inteligencia de Negocios. Procesa comentarios en español, aplica una clasificación de sentimientos basada en un diccionario y reglas de negación, y genera un resumen JSON para el sistema conversacional.

## Estructura

```text
nlp-lab/
├── dataset/
│   └── comentarios_clientes.csv
├── src/
│   └── analizar_sentimientos.py
└── salida/
    ├── comentarios_analizados.csv
    ├── evaluacion_sentimientos.json
    └── resumen_sentimientos.json
```

## Dataset

`dataset/comentarios_clientes.csv` contiene 72 comentarios simulados en español: ocho categorías de producto y nueve comentarios por categoría. Cada categoría incluye tres etiquetas de referencia positivas, tres negativas y tres neutrales.

La columna `sentimiento_referencia` permite comparar el resultado del analizador. No se utiliza para decidir la predicción.

## Requisitos

- Python 3.10 o superior.
- No requiere instalar paquetes externos; utiliza la biblioteca estándar.

## Ejecutar desde la raíz del repositorio

```bash
python nlp-lab/src/analizar_sentimientos.py
```

El programa realiza:
1. Normalización del texto y tokenización.
2. Conteo de términos positivos y negativos.
3. Inversión de polaridad ante negadores cercanos, como «no».
4. Clasificación en positivo, negativo o neutral.
5. Comparación con las etiquetas de referencia.
6. Agrupación de opiniones por producto.

## Archivos generados

- `salida/comentarios_analizados.csv`: texto limpio, tokens, puntajes, sentimiento predicho y referencia.
- `salida/evaluacion_sentimientos.json`: accuracy, F1 macro y métricas por sentimiento.
- `salida/resumen_sentimientos.json`: conteos y comentarios por producto. Es la entrada del sistema conversacional.

## Alcance y limitaciones

Este laboratorio implementa un método léxico basado en reglas; no entrena un modelo de lenguaje ni comprende plenamente ironía, contexto o expresiones implícitas. La evaluación utiliza comentarios simulados y debe interpretarse como una demostración didáctica del método.
