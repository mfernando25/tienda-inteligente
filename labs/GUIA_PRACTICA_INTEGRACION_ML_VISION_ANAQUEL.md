# Cierre de la sesión 1: inferencia del anaquel completo

**Unidad 4 – Inteligencia de Negocios**  
**Ejes 1 y 2:** Machine Learning y Visión por computadora.

## Preparar el entorno y recuperar el trabajo

### Primera vez en una computadora

Desde Git Bash, ubicarse en la carpeta donde se guardarán los proyectos:

~~~bash
git clone https://github.com/mfernando25/tienda-inteligente.git
cd tienda-inteligente/cv-lab
py -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
~~~

### Si el repositorio ya está clonado

Desde la carpeta raíz del repositorio:

~~~bash
git pull
cd cv-lab
source .venv/Scripts/activate
~~~

Si la carpeta `.venv` ya no existe (por ejemplo, después de DeepFreeze), recrearla e instalar las dependencias:

~~~bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
~~~

### Respaldo para continuar después de DeepFreeze

Las carpetas `cv-lab/dataset/` y `cv-lab/salida_ml/` contienen datos y resultados locales: están excluidas de GitHub y no se recuperan con `git clone` ni con `git pull`. Guardar una copia fuera de la zona que DeepFreeze restaura.

Después de clonar o actualizar el repositorio, copiar las carpetas respaldadas a `cv-lab/`, conservando sus nombres y estructura. Según lo que se haya guardado:

- **Se respaldaron `dataset/` y `salida_ml/`:** restaurar ambas carpetas. No es necesario preparar el dataset ni volver a entrenar; se puede continuar con la inferencia.
- **Se respaldó todo `dataset/`, pero no `salida_ml/`:** si están presentes `images/crops/` y `manifests/recortes.csv`, ejecutar `python entrenar_modelo.py` para reconstruir el modelo.
- **Solo se respaldaron las fotografías originales y sus JSON:** ubicarlos en `dataset/images/raw/` y `dataset/labels/`, respectivamente, con el mismo nombre base; ejecutar primero `python preparar_dataset.py` y después `python entrenar_modelo.py`.

Para la inferencia de esta guía deben existir estos dos archivos del modelo:

~~~text
cv-lab/salida_ml/modelo_hog_svm.yml
cv-lab/salida_ml/modelo_hog_svm.json
~~~

No basta con guardar únicamente el código del repositorio: las fotografías, etiquetas, recortes, manifiesto y modelo entrenado son archivos locales.

---

## Punto de partida

En la práctica anterior cada grupo ya:

- Generó 15 fotografías y sus JSON.
- Preparó los 480 recortes con `preparar_dataset.py`.
- Entrenó HOG + SVM con `entrenar_modelo.py`.

**No repetir esos pasos ni volver a entrenar el modelo.** Esta práctica utiliza el modelo que ya está guardado en `cv-lab/salida_ml/`.

## 1. Actualizar el repositorio

Desde Git Bash, en la carpeta raíz del proyecto:

~~~bash
git pull
cd cv-lab
source .venv/Scripts/activate
~~~

El repositorio ya incluye `predecir_anaqueles.py`, que aplica el modelo a un lote de fotografías completas.

## 2. Crear 15 anaqueles nuevos

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

## 3. Ejecutar la clasificación por lotes

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

## 4. Revisar los resultados

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

## 5. Analizar el comportamiento del modelo

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
