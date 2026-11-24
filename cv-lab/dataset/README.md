# Dataset de fotografías del anaquel

Esta carpeta organiza las fotografías Android y las etiquetas de referencia para desarrollar y evaluar la solución de visión por computadora y Machine Learning.

## Estructura y ubicación de los archivos

Todos los archivos de captura se guardan localmente en la laptop del grupo, dentro de `cv-lab/dataset/`. **No se suben las fotografías a GitHub.**

- `images/raw/`: fotografías originales Android (JPG).
- `images/rectified/`: imágenes rectificadas por OpenCV.
- `images/crops/`: recortes automáticos de cada posición.
- `labels/`: JSON exportados por el generador.
- `manifests/`: CSV que registra cada recorte y su etiqueta.

## Evaluación posterior del modelo

Además del dataset inicial de entrenamiento, la evaluación del modelo utiliza cinco fotografías nuevas y sus JSON de referencia. No mezclar estos archivos con las capturas 001–015.

~~~text
iteraciones/
└── evaluacion_01/
    ├── entrada/
    │   ├── captura_031.jpg
    │   └── ... captura_035.jpg
    └── referencias/
        ├── captura_031.json
        └── ... captura_035.json
~~~

Para cada anaquel, exporta **Datos JSON** antes de tomar la fotografía y conserva exactamente la misma distribución. Renombra el JPG y el JSON con el mismo identificador, por ejemplo `captura_031.jpg` y `captura_031.json`.

El clasificador recibe solamente los JPG. Una vez terminadas todas las predicciones, el programa abre los JSON para comparar las etiquetas reales con las predichas y calcular las métricas. Los JSON no intervienen en la inferencia.

## Plan exacto: 15 fotografías de 15 distribuciones diferentes

Cada fotografía debe mostrar una distribución diferente del anaquel. Para cada una debes guardar **dos archivos obligatorios**: la foto JPG de Android y el JSON de esa misma distribución.

| N.º | Qué debes configurar antes de fotografiar | Qué variación se busca |
|---|---|---|
| 001 | Exhibición equilibrada con varias categorías y algunos espacios vacíos. | Primera referencia del conjunto. |
| 002 | Aplicar el escenario «Faltantes de productos». | Una distribución distinta con productos ausentes. |
| 003 | Aplicar «Anaquel completo». | Los 32 espacios ocupados. |
| 004 | Aplicar «Predominio de bebidas». | Mayor presencia de agua, gaseosa y jugo. |
| 005 | Crear una distribución personalizada: cambiar productos y posiciones de varias celdas. | Mezcla variada, con presencia de las ocho categorías. |
| 006 | Crear otra distribución personalizada, diferente de la 005. | Reubicar categorías en otras filas y columnas. |
| 007 | Crear otra distribución personalizada, diferente de las anteriores. | Cambiar cantidades y dejar algunos espacios vacíos. |
| 008 | Crear otra distribución personalizada. | Procurar que productos menos frecuentes aparezcan en varias zonas. |
| 009 | Crear otra distribución personalizada. | Una combinación distinta, procurando representar las ocho categorías. |
| 010 | Crear una distribución con baja ocupación. | Pocos productos agrupados a la izquierda. |
| 011 | Crear otra distribución con baja ocupación, diferente de la 010. | Productos agrupados al centro o a la derecha. |
| 012 | Crear una tercera distribución con baja ocupación. | Productos distribuidos en otra combinación de filas y columnas. |
| 013 | Crear una distribución nueva y activar una interferencia moderada (por ejemplo, cursor sobre una celda). | Mantener visible la cuadrícula y el producto real de cada posición. |
| 014 | Crear otra distribución nueva y añadir una interferencia moderada (por ejemplo, reflejo o brillo desigual). | No repetir la distribución 013. |
| 015 | Crear otra distribución nueva y capturar desde un ángulo algo más oblicuo u ocluir parcialmente una ilustración sin ocultar la cuadrícula. | Probar una condición menos perfecta, pero todavía interpretable. |

Los escenarios predefinidos pueden cambiar las posiciones al pulsar «Aplicar escenario». Comprueba el anaquel y exporta su JSON después de aplicar cada escenario.

Las capturas 005–015 deben tener **configuraciones diferentes**, no basta con tomar otra foto del mismo anaquel. También puede variar moderadamente la distancia o el ángulo, pero la variación principal debe ser la distribución de productos.

### Qué archivos guardar para cada número

Ejemplo para la captura 002:

~~~text
cv-lab/dataset/
├── images/
│   └── raw/
│       └── captura_002.jpg       # foto original tomada con Android
└── labels/
    └── captura_002.json          # JSON exportado para ESA distribución
~~~

Repite el patrón para 001–015. Los nombres base deben coincidir exactamente:

- `images/raw/captura_001.jpg` ↔ `labels/captura_001.json`
- `images/raw/captura_002.jpg` ↔ `labels/captura_002.json`
- …
- `images/raw/captura_015.jpg` ↔ `labels/captura_015.json`

**Obligatorio:** JPG Android + JSON correspondiente.

**No obligatorio:** PNG exportado por el generador. El PNG puede guardarse como referencia visual si ayuda a revisar la distribución, pero OpenCV no lo usa para detectar ni para preparar el dataset. La entrada visual del análisis es el JPG Android; el JSON sirve para etiquetar y comprobar.

### Pasos para producir cada par JPG + JSON

1. Abre el generador en la computadora.
2. Aplica o construye la distribución indicada en la tabla.
3. Pulsa «Datos JSON» y guarda el archivo.
4. Renómbralo con el número de captura, por ejemplo `captura_002.json`.
5. Activa «Modo captura».
6. Fotografía la pantalla con la cámara Android; conserva el JPG original sin filtros.
7. Copia el JPG a `dataset/images/raw/` con el nombre correspondiente, por ejemplo `captura_002.jpg`.
8. Copia el JSON a `dataset/labels/` con el mismo nombre base.
9. Comprueba que el JPG y el JSON pertenecen a la misma distribución antes de continuar.

No cambies la distribución entre exportar el JSON y tomar su fotografía.

### Variaciones e interferencias

Las imperfecciones moderadas se incluyen para probar robustez: cursor, reflejo, brillo desigual, perspectiva u oclusión parcial. La etiqueta debe describir el contenido real:
- Una celda vacía con cursor sigue siendo `empty`.
- Un producto parcialmente tapado conserva su clase si sigue siendo identificable.
- Si no se puede determinar qué producto aparece, no inventes una etiqueta; deja esa muestra para revisión o exclúyela del entrenamiento.

Este anaquel virtual no simula por completo productos ni anaqueles físicos reales.

## Después de guardar las 15 parejas

Desde `cv-lab/`, con el entorno virtual activo, ejecuta una sola vez:

~~~bash
python preparar_dataset.py
~~~

El programa recorre `images/raw/`, busca el JSON del mismo nombre en `labels/`, solicita seleccionar cuatro esquinas por fotografía y genera automáticamente:
- una imagen rectificada por captura;
- 32 recortes por captura, organizados por clase o en `empty/`;
- `manifests/recortes.csv` con la trazabilidad de cada recorte.

No necesitas ejecutar el programa 15 veces. Sí debes seleccionar las cuatro esquinas una vez por cada fotografía.

## Resultado esperado

Con 15 parejas JPG + JSON:
- 15 fotografías originales.
- 15 JSON de referencia.
- Hasta 480 recortes (15 × 32), incluyendo posiciones vacías.
- Un manifiesto con hasta 480 filas de recortes.

Los 480 recortes no son 480 fotografías independientes: los 32 recortes de una captura comparten condiciones de cámara y perspectiva.

## Entrenamiento, validación y prueba

La separación se realizará por fotografía/configuración, no mezclando recortes de una misma captura. Como punto inicial se propone:
- 9 capturas para entrenamiento;
- 3 para validación;
- 3 para prueba final.

Antes de fijar la división comprobaremos que las ocho categorías aparezcan en los tres conjuntos. Si una categoría queda fuera o tiene muy pocos ejemplos, se deben añadir capturas dirigidas. Quince fotografías son un mínimo de partida, no garantía de suficiencia.

## Qué no hacer

- No editar las fotos originales ni aplicar filtros.
- No asociar un JSON de otra distribución.
- No cambiar el anaquel entre exportar el JSON y tomar la foto.
- No repartir recortes de una misma captura entre entrenamiento y prueba.
- No fotografiar personas ni información personal.

Las fotografías y recortes permanecen en la laptop del grupo y están excluidos de GitHub para evitar subir archivos pesados.