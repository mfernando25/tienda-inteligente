# Pautas de continuidad – Unidad 4: Tienda inteligente

Este documento fija las decisiones del proyecto para mantener continuidad entre las sesiones, las guías y el examen. Ante cualquier cambio, actualizar primero este documento y justificar la modificación.

## 1. Propósito común

Desarrollar una solución empresarial didáctica que conecte:

1. Machine Learning y visión por computadora: reconocer productos en un anaquel y estimar su inventario visible.
2. Procesamiento de lenguaje natural: analizar opiniones de clientes.
3. Sistemas conversacionales: consultar inventario y opiniones para responder preguntas del negocio.

No crear ejercicios aislados ni herramientas que no se reutilicen en el proyecto o en el examen.

## 2. Organización de las sesiones

### Sesión 1 – La tienda ve y aprende
**Ejes 1 y 2: Machine Learning y Visión por computadora.**

Estado: los grupos completaron la preparación del dataset y el entrenamiento HOG + SVM. El cierre consiste en evaluar el modelo con cinco fotografías nuevas.

Cierre acordado:
- Crear cinco distribuciones nuevas, capturarlas con Android y exportar el JPG y su JSON de referencia.
- Guardar los JPG y JSON en carpetas separadas, con el mismo nombre base, dentro de `dataset/iteraciones/evaluacion_01/`.
- Aplicar el modelo existente con `predecir_anaqueles.py`, sin reentrenarlo.
- Clasificar primero las 32 posiciones de cada JPG, sin consultar el JSON.
- Cuando terminen todas las predicciones, comparar las clases predichas con las etiquetas reales del JSON.
- Generar un informe HTML con métricas, matriz de confusión, errores y anaqueles anotados.

La guía de Sesión 1 termina con la interpretación del informe de evaluación.

### Sesión 2 – La tienda entiende y conversa
**Ejes 3 y 4: Procesamiento de lenguaje natural y Sistemas conversacionales.**

Tendrá una segunda guía independiente: `labs/GUIA_PRACTICA_PNL_SISTEMAS_CONVERSACIONALES.md`. Reutilizará:
- El inventario visual producido en la Sesión 1: `cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json`.
- El dataset `nlp-lab/dataset/comentarios_clientes.csv`, con 72 comentarios simulados en español, ocho categorías y distribución equilibrada: 24 positivos, 24 negativos y 24 neutrales.

**Eje 3 – PLN y análisis de opiniones**
- Normalización y tokenización exploratoria mediante NLTK.
- Clasificación de sentimientos con un modelo Transformer preentrenado para español, a través de PySentimiento.
- Comparación de las predicciones con `sentimiento_referencia`, con accuracy, F1 macro y matriz de confusión.
- Generación de `nlp-lab/salida/resumen_sentimientos.json` para su uso por el chatbot.
- Interpretación de desacuerdos y limitaciones del dominio del modelo: el modelo de referencia se entrenó principalmente con publicaciones de redes sociales, mientras que la práctica utiliza comentarios comerciales sintéticos.

**Eje 4 – Sistemas conversacionales**
- Clasificación supervisada de intenciones mediante TF-IDF de palabras y caracteres + `LogisticRegression` de scikit-learn. La rama de palabras usa stemming Snowball para español, de modo que formas como producto/productos y predomina/predominan compartan raíz; la rama de caracteres ayuda a reconocer variantes morfológicas y de escritura.
- Dataset de preguntas etiquetadas: `conversational-lab/dataset/intenciones.csv`, con 72 ejemplos iniciales distribuidos en nueve intenciones (ocho por intención). En la práctica, los estudiantes añaden 15 preguntas nuevas: al menos una por intención y las seis restantes distribuidas entre las categorías que necesiten más variedad. Si la puntuación es baja o las dos intenciones más probables son demasiado cercanas, el chatbot solicita reformular en lugar de emitir una respuesta potencialmente incorrecta.
- Validación del clasificador con una partición estratificada del 25% para calcular accuracy y F1 macro; después el asistente se entrena con todos los ejemplos. Las métricas son orientativas por el tamaño reducido del dataset.
- Extracción de entidades de producto mediante alias y consulta de `inventario_por_foto.json` y `resumen_sentimientos.json`.
- Respuestas construidas con plantillas a partir de los datos del proyecto.

El chatbot utiliza aprendizaje automático para identificar intenciones, pero no es generativo ni un LLM. La extracción de entidades y la construcción de respuestas se mantienen determinísticas y limitadas al dominio de la tienda.

Las consultas se plantean como necesidades de negocio: menor y mayor presencia de productos, ausencias en las fotografías, ocupación de anaqueles, inventario visible por categoría y análisis de opiniones/quejas. No se exige recordar códigos de captura. Los cinco JPG representan cinco anaqueles distintos, no un registro temporal del mismo anaquel; por tanto, no permiten inferir cuánto tiempo lleva faltando un producto ni el stock del almacén.

La práctica completa debe ejecutarse en aproximadamente 60 minutos de clase. La instalación de dependencias y la primera descarga del modelo Transformer son preparación previa y no forman parte de esos 60 minutos. PyTorch se instala con la rueda CPU (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`) para evitar depender de GPU NVIDIA; después se instala `nlp-lab/requirements.txt` y se inicializa el modelo de español. Se usa un entorno independiente `.venv-sesion2`, sin modificar el entorno de visión de la Sesión 1. Mantener esta práctica en su segunda guía; no incorporarla en la guía de Sesión 1.

## 3. Iteraciones de evaluación visual

### Evaluación 01 – Cinco anaqueles nuevos

- Entrada: cinco fotografías JPG de anaqueles completos y cinco JSON de referencia con el mismo nombre base.
- Mantener las ocho ilustraciones originales y la clase `empty`.
- Variar las distribuciones para que las nueve clases estén representadas en el conjunto.
- Usar el mismo modelo HOG + SVM ya entrenado.
- No ejecutar `preparar_dataset.py` ni `entrenar_modelo.py` para estas cinco fotografías.
- La clasificación utiliza únicamente el JPG. Los JSON se abren después de completar todas las predicciones y se usan solo para calcular métricas.

El objetivo es evaluar el reconocimiento en nuevas distribuciones y capturas, no inferir el inventario de negocio.

### Evaluación posterior – Nuevos diseños visuales

- Mantener las mismas categorías semánticas e identificadores.
- Cambiar la representación visual de los productos: formas, contornos y detalles, no solamente colores.
- Aplicar inicialmente el mismo modelo, sin reentrenarlo, para observar la pérdida de generalización.
- Conservar los resultados en una carpeta independiente.

Esta prueba estudia las limitaciones del modelo con diseños distintos; no garantiza que reconozca empaques comerciales reales.

## 4. Organización de archivos de evaluación

Las entradas y referencias se guardan separadas de los resultados:

~~~text
cv-lab/
├── dataset/
│   └── iteraciones/
│       └── evaluacion_01/
│           ├── entrada/
│           │   └── captura_031.jpg ... captura_035.jpg
│           └── referencias/
│               └── captura_031.json ... captura_035.json
└── salida_ml/
    └── iteraciones/
        └── evaluacion_01/
            ├── anaqueles_anotados/
            ├── recortes_predichos/
            ├── predicciones.csv
            ├── metricas_por_clase.csv
            ├── matriz_confusion.csv
            ├── errores.csv
            ├── evaluacion.json
            ├── inventario_por_foto.json
            └── informe_evaluacion.html
~~~

Para una evaluación posterior, utilizar una carpeta nueva (por ejemplo, `evaluacion_02`) y mantener separados sus JPG, JSON y resultados.

## 5. Reglas sobre datos y evaluación

- La fase de inferencia recibe fotografías JPG y el modelo entrenado; no consulta JSON.
- La comparación con las etiquetas reales se ejecuta solamente después de terminar todas las predicciones.
- La carpeta de cada recorte se determina por la clase predicha, no por la etiqueta real.
- No corregir ni renombrar manualmente las predicciones.
- Los nombres de archivos conservan el identificador de captura y la posición.
- Las métricas se interpretan como evaluación por posición dentro de cinco anaqueles, no como 160 fotografías independientes.
- El generador representa posiciones (*facings*), no stock físico de almacén.

## 6. Continuidad con el examen final

La solución común del examen reutiliza los componentes trabajados en las dos sesiones. Los seis escenarios inicialmente propuestos son:

1. Productos faltantes.
2. Exceso de existencias.
3. Productos mal ubicados.
4. Inventario con distintas cantidades.
5. Cambios en la demanda.
6. Productos con opiniones negativas.

Cada equipo recibe un escenario y adapta los datos y las reglas del negocio, sin cambiar toda la arquitectura.

Condiciones para que los escenarios sean evaluables:
- Faltantes y exceso requieren umbrales mínimos o máximos definidos.
- Productos mal ubicados requiere un planograma o distribución esperada.
- Cantidades distintas requiere definir si se cuentan posiciones de exhibición o unidades; el generador actual solo representa posiciones.
- Cambios en demanda requiere registros históricos de ventas o demanda.
- Opiniones negativas requiere comentarios etiquetados o una regla de sentimiento definida.
- La dificultad y los entregables deben ser comparables entre equipos.

No asumir que una fotografía proporciona datos de ventas, stock de almacén o satisfacción del cliente.

## 7. Criterios de desarrollo

- Preferir herramientas gratuitas y reproducibles en Windows, sin APIs pagadas ni backend.
- Mantener el entorno de visión de la Sesión 1 separado del entorno `.venv-sesion2`.
- Descargar e inicializar el modelo Transformer antes de la clase para que el tiempo de preparación no consuma los 60 minutos de práctica.
- Mantener el generador actual y sus ocho ilustraciones para la Evaluación 01. Los nuevos diseños visuales se reservan para una prueba posterior de generalización.
- Preparar y probar los scripts con los JSON reales generados por la Sesión 1 antes de llevarlos al aula.
- Las guías deben partir del estado real del repositorio y mostrar comandos, rutas, código y comprobaciones concretas.
- No repetir pasos que los grupos ya completaron.
- Una guía por sesión: la guía de Sesión 1 cierra visión y ML; la guía de Sesión 2 desarrolla PLN y sistemas conversacionales.

## 8. Fuera de alcance por ahora

- Detección automática de anaqueles arbitrarios o de objetos fuera de la cuadrícula 4 × 8.
- Reconocimiento garantizado de empaques comerciales reales.
- Estimación de stock físico detrás de los productos visibles.
- Predicción de demanda sin datos históricos.
- Reentrenamiento durante la Iteración 01.
- Desarrollo de la guía de Sesión 2 dentro de la guía de Sesión 1.
