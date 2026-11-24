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

## Punto de partida

Al inicio de esta guía se prepara el dataset a partir de 15 fotografías JPG y sus JSON, y se entrena el modelo HOG + SVM. Después, se utilizará ese modelo para clasificar 15 anaqueles nuevos.

El repositorio incluye `predecir_anaqueles.py`, que aplica el modelo a un lote de fotografías completas.

## 1. Crear 15 anaqueles nuevos

Iniciar el generador:

~~~bash
cd ../shelf-generator
python -m http.server 8000
~~~

Abrir http://localhost:8000.

Crear **15 distribuciones diferentes** usando las mismas ocho ilustraciones de productos del generador. No repetir las distribuciones de las capturas 001–015.

- Cambiar la ubicación y cantidad de productos entre fotografías.
- Incluir anaqueles con ocupación alta, media y baja.
- Variar moderadamente la distancia, el ángulo o la iluminación en algunas capturas.
- Mantener visibles las cuatro filas, las ocho columnas y las esquinas exteriores.
- Tomar cada fotografía completa con Android, en horizontal, enfocada y sin filtros.

Guardar los 15 archivos JPG en:

~~~text
cv-lab/dataset/iteraciones/iteracion_01/entrada/
├── captura_016.jpg
├── captura_017.jpg
├── ...
└── captura_030.jpg
~~~

**Para esta iteración no se necesita exportar ni guardar el JSON del generador.** El clasificador debe asignar las categorías únicamente a partir de las fotografías. El programa sí consulta `salida_ml/modelo_hog_svm.json`, pero ese archivo contiene la configuración y las clases del modelo; no contiene las etiquetas de estos nuevos anaqueles.

Detener el servidor con `Ctrl + C` y volver a `cv-lab/`:

~~~bash
cd ../cv-lab
~~~

## 2. Ejecutar la clasificación por lotes

Ejecutar:

~~~bash
python predecir_anaqueles.py
~~~

El programa procesa cada fotografía. Para cada una, seleccionar las cuatro esquinas exteriores de la cuadrícula en este orden:

1. Superior izquierda.
2. Superior derecha.
3. Inferior derecha.
4. Inferior izquierda.

Pulsar Enter para confirmar los cuatro puntos. Repetir en cada fotografía.

El programa utiliza el modelo existente y clasifica las 32 posiciones. **No ejecuta entrenamiento ni consulta etiquetas del generador.**

Si se necesita repetir la misma iteración y reemplazar sus resultados:

~~~bash
python predecir_anaqueles.py --sobrescribir
~~~

## 3. Revisar los resultados

El programa crea automáticamente:

~~~text
cv-lab/salida_ml/iteraciones/iteracion_01/
├── anaqueles_anotados/
├── recortes_predichos/
│   ├── canned/
│   ├── cereal/
│   ├── cookies/
│   ├── detergent/
│   ├── empty/
│   ├── juice/
│   ├── milk/
│   ├── soda/
│   └── water/
├── predicciones.csv
├── inventario_por_foto.csv
└── inventario_por_foto.json
~~~

- `anaqueles_anotados/`: cada anaquel con sus 32 predicciones.
- `recortes_predichos/`: cada recorte guardado en la carpeta de la clase que predijo el modelo. El nombre conserva la captura y la posición.
- `predicciones.csv`: una fila por posición clasificada.
- `inventario_por_foto.csv` y `inventario_por_foto.json`: conteo estimado por categoría para cada anaquel.

**Cada fotografía representa un anaquel independiente.** No sumar las cantidades de las 15 fotografías como si fueran un único inventario físico.

## 4. Analizar el comportamiento del modelo

Revisar las imágenes anotadas y las carpetas de recortes. Responder:

1. ¿Qué categorías predijo con mayor frecuencia?
2. ¿Qué categorías aparecen poco o no aparecen en las predicciones?
3. ¿Se observan recortes colocados en carpetas que no corresponden a su producto?
4. ¿Qué condiciones de captura parecen afectar más al reconocimiento?
5. ¿Qué productos podrían requerir reposición según su presencia en el anaquel?

Para la última pregunta, utilizar los mínimos de exhibición que indique el caso de negocio. Sin un umbral definido, reportar las cantidades visibles, pero no afirmar que existe una necesidad de reposición.

## Entregables

- Las 15 fotografías nuevas, de `captura_016.jpg` a `captura_030.jpg`.
- La carpeta `iteracion_01/` con los anaqueles anotados y recortes clasificados.
- `predicciones.csv` e `inventario_por_foto.csv`.
- Una conclusión breve sobre los aciertos, errores y limitaciones observadas.

No subir fotografías ni resultados locales al repositorio público. El código fuente sí puede versionarse.
