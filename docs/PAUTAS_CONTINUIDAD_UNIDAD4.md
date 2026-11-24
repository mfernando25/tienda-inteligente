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

Estado: los grupos completaron la preparación del dataset y el entrenamiento HOG + SVM. Falta cerrar la integración de inferencia sobre anaqueles completos.

Cierre acordado:
- Recibir 15 fotografías JPG nuevas de anaqueles completos, capturadas con Android, en `dataset/iteraciones/iteracion_01/entrada/`.
- Mantener las ocho categorías e ilustraciones del generador actual, variando las distribuciones y condiciones de captura.
- Aplicar el modelo existente con `predecir_anaqueles.py`, sin reentrenarlo.
- No usar JSON del generador como entrada de predicción.
- Clasificar las 32 posiciones de cada fotografía.
- Guardar cada recorte en una carpeta según la clase predicha.
- Generar la imagen anotada y el inventario estimado por fotografía.

La guía de Sesión 1 termina cuando se procesa el lote, se revisan las predicciones y se interpreta el inventario visual.

### Sesión 2 – La tienda entiende y conversa
**Ejes 3 y 4: Procesamiento de lenguaje natural y Sistemas conversacionales.**

Tendrá una guía independiente. Reutilizará:
- El inventario visual producido en la Sesión 1.
- Un conjunto de comentarios de clientes sobre las categorías de productos.

El trabajo de PLN abarcará limpieza, tokenización y análisis de sentimientos. El sistema conversacional identificará intenciones y entidades y consultará las fuentes estructuradas de inventario y opiniones.

No incorporar esta práctica dentro de la guía de Sesión 1.

## 3. Iteraciones de evaluación visual

### Iteración 01 – Mismos diseños, nuevos anaqueles

- Entrada: 15 fotografías JPG de anaqueles completos, no fotos individuales de productos.
- Mantener las ocho ilustraciones originales y la clase `empty`.
- Variar la distribución de productos y las condiciones moderadas de captura.
- Usar el mismo modelo HOG + SVM ya entrenado.
- No ejecutar `preparar_dataset.py` ni `entrenar_modelo.py`.
- No usar JSON como entrada ni ayuda para la predicción.

El JSON no es necesario para clasificar. Si se conserva una copia como referencia para una comparación posterior, debe permanecer separada del flujo de inferencia y no será requisito para generar los resultados.

### Iteración 02 – Nuevos diseños visuales

- Mantener las mismas categorías semánticas e identificadores.
- Cambiar la representación visual de los productos: formas, contornos y detalles, no solamente colores.
- Aplicar inicialmente el mismo modelo, sin reentrenarlo, para observar la pérdida de generalización.
- Conservar los resultados en una carpeta independiente.

Esta iteración es una prueba de limitaciones, no una promesa de que el modelo reconocerá empaques o diseños desconocidos. Una mejora o reentrenamiento posterior debe definirse como una actividad distinta.

## 4. Organización de resultados por iteración

Cada iteración mantiene sus propios archivos. No sobrescribir ni mezclar resultados.

Estructura acordada:

~~~text
cv-lab/
└── salida_ml/
    └── iteraciones/
        ├── iteracion_01/
        │   ├── entrada/
        │   ├── anaqueles_anotados/
        │   ├── recortes_predichos/
        │   │   ├── canned/
        │   │   ├── cereal/
        │   │   ├── cookies/
        │   │   ├── detergent/
        │   │   ├── empty/
        │   │   ├── juice/
        │   │   ├── milk/
        │   │   ├── soda/
        │   │   └── water/
        │   ├── predicciones.csv
        │   └── inventario_por_foto.csv
        └── iteracion_02/
            └── misma estructura
~~~

Cada recorte conserva el identificador de captura y la posición, por ejemplo `captura_016_slot_05.png`.

Cada anaquel es un escenario independiente. El inventario debe presentarse por fotografía; no sumar las 15 fotografías como si fueran un único inventario físico. El CSV de predicciones conserva una fila por posición y permite rastrear cada clasificación.

## 5. Reglas sobre datos y evaluación

- La inferencia recibe fotografías JPG y el modelo entrenado; no recibe JSON.
- La carpeta de destino de cada recorte se determina exclusivamente por la predicción del modelo.
- No cambiar, corregir ni renombrar manualmente las clases predichas para hacer parecer mejor el resultado.
- Los nombres de archivo deben conservar captura y posición.
- El JSON solo puede utilizarse como referencia externa posterior, si se decide calcular métricas contra la distribución real.
- Distinguir inventario visual de posiciones (*facings*) de stock físico o unidades en almacén. El generador actual muestra un producto por posición y no permite inferir unidades detrás del frente.

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

- Preferir herramientas locales, gratuitas y reproducibles en Windows.
- No añadir APIs pagadas, backend o dependencias que no sean necesarias.
- Mantener el generador actual para la Iteración 01. Los nuevos diseños se incorporan solamente en la Iteración 02.
- Preparar y probar los scripts antes de llevarlos al aula.
- Las guías deben partir del estado real del repositorio y mostrar comandos, rutas, código y comprobaciones concretas.
- No repetir pasos que los grupos ya completaron.
- Una guía por sesión: la guía de Sesión 1 cierra visión y ML; la guía de Sesión 2 desarrolla PLN y conversación.

## 8. Fuera de alcance por ahora

- Detección automática de anaqueles arbitrarios o de objetos fuera de la cuadrícula 4 × 8.
- Reconocimiento garantizado de empaques comerciales reales.
- Estimación de stock físico detrás de los productos visibles.
- Predicción de demanda sin datos históricos.
- Reentrenamiento durante la Iteración 01.
- Desarrollo de la guía de Sesión 2 dentro de la guía de Sesión 1.
