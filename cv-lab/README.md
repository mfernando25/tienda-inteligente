# OpenCV y Machine Learning: visión del anaquel

Esta carpeta contiene la primera etapa de visión por computadora y el primer clasificador de productos del proyecto.

## 1. Detección inicial de ocupación

El programa analizar_anaquel.py recibe una fotografía Android, permite seleccionar las cuatro esquinas de la cuadrícula 4×8, corrige la perspectiva y divide el anaquel en 32 posiciones. Después utiliza umbrales HSV (saturación, brillo y proporción de color) para clasificar cada posición como ocupada o vacía.

**Esta detección inicial usa reglas fijas: no es Machine Learning y no identifica las ocho categorías de productos.**

Desde cv-lab:

~~~bash
python analizar_anaquel.py "dataset/images/raw/captura_001.jpg"
~~~

Se deben seleccionar las esquinas en orden: superior izquierda, superior derecha, inferior derecha e inferior izquierda.

Parámetros disponibles:
- --saturacion-min: saturación HSV mínima (valor inicial 65).
- --brillo-min: brillo HSV mínimo (valor inicial 45).
- --ratio-min: proporción mínima de píxeles de color (valor inicial 0.08).

Cambiar estos umbrales ajusta las reglas, no entrena un modelo.

## 2. Preparar las fotografías y etiquetas

Los archivos de captura se guardan localmente dentro de dataset/ y no se suben a GitHub:

- images/raw/: fotografías Android originales.
- images/rectified/: fotografías rectificadas.
- images/crops/: recortes de las 32 posiciones.
- labels/: JSON exportados por el generador.
- manifests/: CSV de trazabilidad de los recortes.

Cada foto necesita su JSON correspondiente con el mismo nombre base. Ejemplo:

~~~text
dataset/images/raw/captura_001.jpg
dataset/labels/captura_001.json
~~~

El JPG Android es la entrada visual. El JSON proporciona la etiqueta de referencia para cada posición; no se usa como entrada visual para reconocer el producto. El PNG exportado por el generador es opcional.

Consulta dataset/README.md para el plan de capturas 001–015 y las variaciones previstas.

## 3. Generar los recortes etiquetados

Después de guardar las parejas JPG + JSON, ejecutar una sola vez desde cv-lab:

~~~bash
python preparar_dataset.py
~~~

El programa procesa cada fotografía, solicita las cuatro esquinas, rectifica el anaquel, recorta las 32 posiciones y guarda cada recorte en la carpeta indicada por el product_id del JSON. La posición sin producto se guarda en empty/. También genera manifests/recortes.csv.

El etiquetado se obtiene del JSON: el programa todavía no aprende a reconocer cereal, leche, agua u otras categorías.

## 4. Entrenar el primer modelo HOG + SVM

Desde cv-lab, con el entorno virtual activado:

~~~bash
python entrenar_modelo.py
~~~

El script realiza lo siguiente:

1. Lee manifests/recortes.csv y verifica las rutas de los recortes.
2. Separa por fotografía completa para evitar que recortes de una misma captura aparezcan en más de un conjunto:
   - Capturas 001–009: entrenamiento.
   - Capturas 010–012: validación.
   - Capturas 013–015: prueba final.
3. Extrae características HOG de cada recorte en escala de grises. HOG representa bordes y orientaciones de gradiente.
4. Entrena un SVM multiclase lineal con las muestras de entrenamiento. Las clases se obtienen de los identificadores del manifiesto; normalmente son ocho productos y empty.
5. Evalúa con validación y prueba, sin usar estas imágenes para entrenar el SVM.

El programa se detiene si falta alguna categoría en entrenamiento. Si una categoría no aparece en validación o prueba, emite una advertencia: no se puede medir su rendimiento en ese conjunto.

La división 9/3/3 conserva la planificación original de variaciones: las capturas de prueba 013–015 contienen condiciones menos favorables. Es una primera evaluación controlada, no una garantía de generalización.

### Informe visual para estudiantes

Al finalizar se genera **salida_ml/informe_evaluacion.html**. Ábrelo con Chrome o Edge. Presenta:
- Indicadores generales de prueba (accuracy, F1 macro, F1 ponderado y cantidad de errores).
- Tabla de precision, recall y F1 por categoría.
- Matriz de confusión visual: diagonal = aciertos; fuera de la diagonal = confusiones.
- Galería de los recortes mal clasificados, con etiqueta real y predicción del modelo.
- Contexto de la división y limitaciones de la evaluación.

Este HTML es el informe de lectura para la práctica. Los CSV se conservan como archivos de datos para análisis adicional, no como interfaz principal para estudiantes.

### Archivos de salida

Se guardan localmente en salida_ml/ (excluida de GitHub):

- modelo_hog_svm.yml: clasificador entrenado.
- modelo_hog_svm.json: clases y configuración del modelo.
- resultados.json: resumen de las métricas.
- matriz_confusion_validation.csv y matriz_confusion_test.csv.
- metricas_por_clase_validation.csv y metricas_por_clase_test.csv.
- errores_validation.csv y errores_test.csv: recortes que el modelo clasificó incorrectamente.

Métricas:
- Accuracy: proporción total de predicciones correctas.
- Precision: de las predicciones asignadas a una clase, cuántas fueron correctas.
- Recall: de los ejemplos reales de una clase, cuántos identificó.
- F1: equilibrio entre precision y recall.
- Matriz de confusión: muestra qué categorías confunde el modelo.

El valor de prueba es la referencia principal de esta primera versión. No se debe ajustar el modelo mirando repetidamente el conjunto de prueba; los cambios se deben orientar con entrenamiento y validación.

## 5. Comparar la detección de ocupación con el JSON

Esta evaluación independiente compara ocupado/vacío, no las ocho categorías:

~~~bash
python comparar_referencia.py --detecciones salida_cv/detecciones.csv --referencia "dataset/labels/captura_001.json"
~~~

## 6. Limitaciones conocidas

- Las ocho clases usan ilustraciones controladas del generador, no empaques comerciales reales.
- HOG + SVM constituye una línea base clásica y ligera; puede confundir productos con formas o diseños parecidos.
- Los 480 recortes no son 480 fotografías independientes: los 32 recortes de una captura comparten iluminación, perspectiva y condiciones de cámara.
- La siguiente fase, después de evaluar esta línea base, será construir un generador independiente con nuevos diseños visuales para probar qué tanto generaliza el modelo.


## 7. Inferencia y evaluación con nuevas fotografías

El programa `predecir_anaqueles.py` aplica el modelo HOG + SVM a fotografías nuevas y, al terminar todas las predicciones, las compara con los JSON de referencia.

### Estructura de entrada

~~~text
dataset/iteraciones/evaluacion_01/
├── entrada/
│   ├── captura_031.jpg
│   ├── ...
│   └── captura_035.jpg
└── referencias/
    ├── captura_031.json
    ├── ...
    └── captura_035.json
~~~

Cada JPG debe tener un JSON de la misma distribución y con el mismo nombre base. El JSON no se utiliza para clasificar: se abre únicamente después de completar las predicciones.

### Ejecutar

Desde `cv-lab/`, con el entorno virtual activado:

~~~bash
python predecir_anaqueles.py
~~~

El programa solicita seleccionar las cuatro esquinas de cada cuadrícula 4×8. Después clasifica las 32 posiciones por fotografía y compara las clases predichas con las etiquetas reales del JSON.

### Resultados

Se guardan en `salida_ml/iteraciones/evaluacion_01/`:

- `informe_evaluacion.html`: informe visual con accuracy, precision, recall, F1, matriz de confusión, errores y anaqueles anotados.
- `predicciones.csv`: detalle de cada posición, clase real y predicha.
- `metricas_por_clase.csv`: métricas por categoría.
- `matriz_confusion.csv`: valores de la matriz.
- `errores.csv`: posiciones mal clasificadas.
- `evaluacion.json`: resumen estructurado.
- `anaqueles_anotados/` y `recortes_predichos/`: imágenes de salida.

Abrir el informe desde Windows:

~~~bash
explorer.exe salida_ml/iteraciones/evaluacion_01/informe_evaluacion.html
~~~

Para evaluar otro lote, indicar nuevas carpetas de entrada, referencias y salida. Por ejemplo:

~~~bash
python predecir_anaqueles.py --entrada dataset/iteraciones/evaluacion_02/entrada --referencias dataset/iteraciones/evaluacion_02/referencias --salida salida_ml/iteraciones/evaluacion_02
~~~

Si se vuelve a ejecutar una evaluación en una carpeta de salida que ya contiene resultados, añadir `--sobrescribir`. Esto elimina y genera nuevamente esa carpeta de resultados.

La evaluación es exploratoria: las posiciones de una misma fotografía comparten condiciones de captura y no son observaciones independientes. El modelo se prueba con las ilustraciones del generador, no con empaques comerciales reales.
