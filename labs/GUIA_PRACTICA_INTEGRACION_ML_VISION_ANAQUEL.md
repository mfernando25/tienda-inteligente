# Cierre de la sesión 1: inferencia del anaquel completo

**Unidad 4 – Inteligencia de Negocios**  
**Ejes 1 y 2:** Machine Learning y Visión por computadora.

## Preparar el entorno

### 1. Descargar el proyecto e instalar dependencias

Si es la primera vez, desde Git Bash:

~~~bash
git clone https://github.com/mfernando25/tienda-inteligente.git
cd tienda-inteligente/cv-lab
py -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
~~~

Si el repositorio ya está clonado, desde su carpeta raíz:

~~~bash
git pull
cd cv-lab
~~~

Si la carpeta `.venv/` no existe en esa computadora, crearla. Después, activar el entorno e instalar las dependencias:

~~~bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
~~~

Si `.venv/` ya existe, basta con activarla:

~~~bash
source .venv/Scripts/activate
~~~

### 2. Copiar las fotografías y sus etiquetas

Copiar las 15 fotografías JPG originales en `cv-lab/dataset/images/raw/` y sus 15 JSON en `cv-lab/dataset/labels/`.

Cada pareja debe conservar el mismo nombre base:

~~~text
dataset/images/raw/captura_001.jpg
dataset/labels/captura_001.json
~~~

Repetir el patrón hasta `captura_015`. No copiar aquí las fotografías nuevas de la iteración 01.

### 3. Preparar el dataset y entrenar el modelo

Desde `cv-lab/`, ejecutar:

~~~bash
python preparar_dataset.py
python entrenar_modelo.py
~~~

El primer programa solicita seleccionar las cuatro esquinas de cada anaquel y genera los 480 recortes etiquetados. El segundo entrena y evalúa HOG + SVM y guarda el modelo en `salida_ml/`.

Al finalizar, deben existir:

~~~text
salida_ml/modelo_hog_svm.yml
salida_ml/modelo_hog_svm.json
~~~

Con esto queda listo el modelo que se utilizará en la Iteración 01.

---

## Evaluar el modelo con cinco anaqueles nuevos

El objetivo ahora es comprobar si el modelo reconoce correctamente productos en fotografías nuevas. Para ello se comparan sus predicciones con los JSON de referencia del generador.

**El modelo solo recibe los JPG. Los JSON se consultan después de finalizar todas las predicciones, exclusivamente para calcular las métricas.**

### 1. Crear cinco anaqueles nuevos

Desde `cv-lab/`, iniciar el generador:

~~~bash
cd ../shelf-generator
python -m http.server 8000
~~~

Abrir http://localhost:8000 y crear cinco distribuciones diferentes con las mismas ocho ilustraciones. Procurar que, entre las cinco, aparezcan las ocho categorías y espacios vacíos.

Distribuciones sugeridas:

| Captura | Configuración |
|---|---|
| 031 | Exhibición equilibrada. |
| 032 | Faltantes de productos. |
| 033 | Anaquel completo. |
| 034 | Predominio de bebidas. |
| 035 | Distribución personalizada, diferente de las anteriores. |

Para **cada distribución**, realizar este orden:

1. Aplicar o construir la distribución en el generador.
2. Pulsar **Datos JSON** y descargar el archivo.
3. Renombrar el JSON con el identificador de la captura, por ejemplo `captura_031.json`.
4. Activar **Modo captura** y tomar la fotografía completa con Android.
5. Guardar el JPG como `captura_031.jpg`.
6. No modificar la distribución entre la exportación del JSON y la fotografía.

Tomar las fotos en horizontal, enfocadas, con las cuatro filas, ocho columnas y esquinas exteriores visibles. Se puede variar moderadamente el ángulo o la iluminación.

Guardar las fotografías en:

~~~text
cv-lab/dataset/iteraciones/evaluacion_01/entrada/
├── captura_031.jpg
├── captura_032.jpg
├── captura_033.jpg
├── captura_034.jpg
└── captura_035.jpg
~~~

Guardar los JSON correspondientes en:

~~~text
cv-lab/dataset/iteraciones/evaluacion_01/referencias/
├── captura_031.json
├── captura_032.json
├── captura_033.json
├── captura_034.json
└── captura_035.json
~~~

**El JPG y el JSON deben tener exactamente el mismo nombre base.** No se necesita guardar el PNG exportado por el generador.

Detener el servidor con `Ctrl + C` y volver a `cv-lab/`:

~~~bash
cd ../cv-lab
~~~

### 2. Ejecutar predicción y evaluación

Con el entorno virtual activado, ejecutar:

~~~bash
python predecir_anaqueles.py
~~~

Para cada JPG, seleccionar las cuatro esquinas exteriores de la cuadrícula 4×8 en este orden:

1. Superior izquierda.
2. Superior derecha.
3. Inferior derecha.
4. Inferior izquierda.

Pulsar Enter para confirmar los cuatro puntos. Repetir en las cinco fotografías.

El programa trabaja en dos fases:

1. Clasifica las 32 posiciones de cada JPG usando el modelo entrenado.
2. Cuando termina todas las predicciones, abre los JSON correspondientes y compara las clases predichas con las reales.

Los JSON **no intervienen en la clasificación**. Se utilizan únicamente para evaluar el resultado.

### 3. Revisar el informe

El programa guarda los resultados en:

~~~text
cv-lab/salida_ml/iteraciones/evaluacion_01/
├── anaqueles_anotados/
├── recortes_predichos/
├── predicciones.csv
├── metricas_por_clase.csv
├── matriz_confusion.csv
├── errores.csv
├── evaluacion.json
└── informe_evaluacion.html
~~~

Abrir el informe en el navegador:

~~~bash
explorer.exe salida_ml/iteraciones/evaluacion_01/informe_evaluacion.html
~~~

El informe presenta:

- **Accuracy:** proporción de posiciones clasificadas correctamente.
- **Precisión, recall y F1:** rendimiento por categoría.
- **Matriz de confusión:** categorías que el modelo reconoce y aquellas que confunde.
- **Galería de errores:** recortes cuya clase real no coincide con la predicción.
- **Anaqueles anotados:** vista de las cinco fotografías con las clases predichas.

Los CSV y el JSON son archivos de datos complementarios; no es necesario abrirlos para realizar el análisis principal.

### 4. Interpretar los resultados

Responder a partir del informe:

1. ¿Qué categorías reconoce mejor el modelo?
2. ¿Qué categorías confunde con mayor frecuencia?
3. ¿Qué muestra la matriz de confusión sobre esos errores?
4. ¿Los errores parecen relacionarse con la forma del producto, el recorte, la perspectiva o la iluminación?
5. ¿Qué limitaciones tiene esta evaluación?

**Alcance:** se evalúan 160 posiciones pertenecientes a cinco fotografías. Las posiciones de una misma fotografía comparten condiciones de captura y no son observaciones independientes. Esta prueba es exploratoria y utiliza las mismas ilustraciones controladas del generador; no demuestra reconocimiento de empaques comerciales reales.

## Entregables

- Cinco fotografías JPG y sus cinco JSON de referencia.
- `informe_evaluacion.html`.
- Una conclusión breve sobre el rendimiento y las limitaciones del modelo.

No subir fotografías, JSON de referencia ni resultados locales al repositorio público. El código fuente sí puede versionarse.
