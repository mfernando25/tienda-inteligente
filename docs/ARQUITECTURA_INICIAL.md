# Arquitectura inicial del proyecto

> **Estado y decisiones vigentes:** este documento conserva el planteamiento inicial. Para la secuencia actual de sesiones, las iteraciones de evaluación, los límites técnicos y la continuidad con el examen, consultar [PAUTAS_CONTINUIDAD_UNIDAD4.md](PAUTAS_CONTINUIDAD_UNIDAD4.md). En caso de diferencia, prevalece ese documento de continuidad.

## Contexto

El proyecto se desarrollará para la Unidad 4 de Inteligencia de Negocios. Los equipos trabajarán en grupos de 4 a 7 estudiantes y mantendrán los mismos integrantes durante los dos laboratorios y el examen práctico.

Cada clase dispone de 3 horas: 90 minutos de teoría y 90 minutos de laboratorio.

## Restricciones conocidas

- Las PC del laboratorio tienen Windows y Deep Freeze; los cambios locales no persisten al reiniciar.
- Al menos una persona por grupo dispone de una laptop con Windows, Python y Git instalado. Ese equipo será el espacio persistente del grupo.
- Los celulares utilizados para la captura serán Android.
- La red Wi-Fi académica de los celulares se considera aislada de la red cableada de las PC del laboratorio.
- No se diseñará transferencia directa celular–PC entre esas redes.
- Se considera un almacenamiento compartido accesible por Internet como posible puente para transferir imágenes. La disponibilidad del servicio aún debe comprobarse.
- El procesamiento se priorizará localmente.
- No se dependerá de APIs ni servicios externos de pago.

## Primer componente: generador de anaqueles

### Alcance acordado

- Aplicación web 2D sencilla que muestra un único anaquel.
- Distribución de 4 filas por 8 columnas (32 posiciones).
- Catálogo inicial de ocho productos ilustrados mediante recursos SVG locales.
- Seleccionar una posición y asignar, reemplazar o retirar el producto.
- Escenarios controlados: equilibrado, faltantes, completo y predominio de bebidas.
- Generación aleatoria adicional y opción de vaciar el anaquel.
- Resumen: posiciones ocupadas, vacías, variedad de productos, ocupación y valor referencial de exhibición.
- Modo captura que oculta los controles para fotografiar desde Android.
- Exportación PNG de referencia, JSON con metadatos del escenario y CSV con una fila por posición.
- Sin backend, base de datos, frameworks ni dependencias externas.

### Implementación disponible

Ubicación: `shelf-generator/`.

En Windows:
1. Abrir PowerShell en `shelf-generator`.
2. Ejecutar `python -m http.server 8000`.
3. Abrir `http://localhost:8000`.

### Semántica de los datos

- Cada celda representa una posición de exhibición (*facing*).
- Un espacio ocupado se cuenta como un producto visible para el cálculo referencial.
- El valor referencial suma el precio unitario ilustrativo por cada espacio ocupado.
- Esto **no equivale al inventario real ni al stock total de almacén**.
- Los archivos JSON y CSV son los datos de referencia para comparar las futuras detecciones visuales (*ground truth*).

### Límites del prototipo

- Las ilustraciones son representaciones controladas, no empaques reales.
- El catálogo contiene ocho clases conocidas. El prototipo no reconoce productos arbitrarios.
- No se ha incorporado OpenCV ni un modelo de ML.
- Los escenarios no se guardan automáticamente; exportar JSON/CSV para conservarlos.
- El modo captura solo oculta controles; la foto se toma en Android.
- La transferencia Android → almacenamiento compartido → laptop continúa pendiente de prueba.

## Visión por computadora: primera etapa implementada

Ubicación: cv-lab/.

La primera implementación recibe únicamente la fotografía Android como entrada del análisis. El usuario selecciona las cuatro esquinas de la cuadrícula 4×8; OpenCV aplica una transformación de perspectiva, divide la imagen en 32 posiciones y detecta si cada una está ocupada o vacía mediante un umbral de saturación en HSV.

Archivos generados:
- Imagen del anaquel rectificado.
- Imagen anotada con detecciones ocupado/vacío.
- CSV con una fila por posición.
- JSON con parámetros y resultados.

Un script independiente compara el CSV con el JSON de referencia exportado por el generador. Ese JSON **no interviene en la detección**, solo en la evaluación posterior.

### Alcance y limitaciones actuales

- La detección se basa en las ilustraciones de color y en la geometría del generador actual.
- No identifica aún las ocho clases de productos.
- No es un modelo de ML ni un detector genérico.
- Los umbrales deben evaluarse con otras fotografías y dispositivos.
- La coincidencia con una única fotografía no demuestra generalización.

## Organización del dataset y situación del aprendizaje

Ubicación: cv-lab/dataset/.

Estructura:
- images/raw/: originales Android.
- images/rectified/: imágenes corregidas por perspectiva.
- images/crops/: recortes de las 32 posiciones.
- labels/: JSON de referencia emparejados por fotografía.

Las fotos originales y las imágenes procesadas se mantienen localmente y se excluyen de Git para evitar repositorios pesados. Cada foto se empareja con su JSON mediante un nombre base común.

### Diferencia entre la primera regla y el futuro modelo ML

La versión actual de analizar_anaquel.py no aprende de imágenes. Detecta ocupado/vacío usando umbrales HSV configurables:
- saturación mínima: 65;
- brillo mínimo: 45;
- proporción mínima de píxeles de color: 0.08.

Los estudiantes pueden modificar esos parámetros y comparar resultados. Eso es ajuste de reglas, no entrenamiento ML.

Las fotografías múltiples se recopilarán para:
1. probar la robustez de la detección ante cambios de dispositivo, distancia, ángulo y luz;
2. rectificar y recortar posiciones;
3. asociar cada recorte con la clase del JSON;
4. formar conjuntos separados de entrenamiento, validación y prueba para clasificar los ocho productos.

Todavía no se ha implementado el recorte etiquetado ni el entrenamiento del clasificador.

## Captura y preparación del dataset

Se acordó un mínimo de **15 fotografías completas por grupo**, tomadas con Android, acompañadas de un JSON de referencia por captura.

Plan aplicado y detallado en cv-lab/dataset/README.md: 15 fotografías completas de 15 distribuciones diferentes. Capturas 001–004: cuatro escenarios predefinidos; 005–009: cinco distribuciones personalizadas; 010–012: baja ocupación/productos concentrados; 013–015: distribuciones con interferencias visuales moderadas. Cada captura requiere un JPG Android y su propio JSON, ambos con el mismo nombre base y correspondientes exactamente a la misma distribución. El PNG del generador es opcional y no se procesa.

Se incorporó \`cv-lab/preparar_dataset.py\`. En una sola ejecución:
1. recorre todas las fotografías de \`dataset/images/raw/\`;
2. busca el JSON correspondiente en \`dataset/labels/\`;
3. solicita seleccionar las cuatro esquinas de cada fotografía;
4. rectifica el anaquel;
5. extrae automáticamente las 32 regiones;
6. asigna las etiquetas usando el JSON;
7. organiza los recortes por clase y genera \`dataset/manifests/recortes.csv\`.

En esta versión se seleccionan cuatro esquinas manualmente **una vez por fotografía**. La automatización de esa localización queda como mejora futura.

Los recortes están organizados en ocho carpetas de producto y una carpeta \`empty/\` para posiciones vacías. Se obtienen hasta 480 recortes por grupo con 15 fotografías; no son 480 imágenes independientes, ya que 32 proceden de la misma captura. La separación inicial será por captura (9 entrenamiento, 3 validación y 3 prueba), comprobando antes que las ocho clases estén representadas en cada conjunto. Quince capturas son el mínimo de partida, no una garantía de suficiencia.

## Separación entre práctica y evaluación

La intención es reutilizar la arquitectura del generador, pero diferenciar el catálogo y los requerimientos del laboratorio y del examen práctico. El generador permite crear escenarios de supermercado de práctica; la evaluación podrá utilizar un catálogo diferente, siempre que el método de visión y las clases se preparen y validen antes.

No se debe asumir que un modelo de visión reconocerá automáticamente productos nuevos.

## Pendientes técnicos y didácticos

1. Probar el generador en Windows y navegadores disponibles.
2. Verificar que PNG, JSON y CSV se descarguen correctamente y que los conteos coincidan.
3. Probar la legibilidad del anaquel al fotografiar una pantalla con distintos celulares Android.
4. Definir el proceso de transferencia de fotografías mediante almacenamiento compartido accesible desde ambas redes.
5. Evaluar distorsión de perspectiva, reflejos, resolución y patrones de pantalla.
6. Elegir un método de visión por computadora realista para 90 minutos y comprobarlo con las fotografías.
7. Definir las clases y datos de entrenamiento/prueba. El JSON/CSV del escenario permitirá contrastar la detección.
8. Decidir cómo ML utilizará los resultados visuales para producir una clasificación o predicción empresarial.
9. Diseñar PLN y sistemas conversacionales para consumir datos del mismo caso.

## Principio de desarrollo

Construir en incrementos pequeños y verificables. Primero validar el generador y sus exportaciones con dispositivos reales; después integrar visión por computadora, ML, PLN y sistemas conversacionales.
