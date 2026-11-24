#!/usr/bin/env python3
"""Clasifica anaqueles y evalúa las predicciones contra JSON de referencia.

El clasificador recibe únicamente las fotografías. Los JSON se abren después
de completar todas las predicciones y se usan exclusivamente para evaluarlas.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from analizar_anaquel import (
    COLUMNS,
    ROWS,
    CELL_HEIGHT,
    CELL_WIDTH,
    read_image,
    rectify_shelf,
    save_image,
    select_corners,
)
from preparar_dataset import crop_slot, load_labels

BASE = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_NAMES_ES = {
    "canned": "Conservas",
    "cereal": "Cereal",
    "cookies": "Galletas",
    "detergent": "Detergente",
    "empty": "Vacío",
    "juice": "Jugo",
    "milk": "Leche",
    "soda": "Gaseosa",
    "water": "Agua",
}


def resolve_from_project(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (BASE / path).resolve()


def load_classifier(model_path: Path, metadata_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No se encontró la configuración del modelo: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = metadata.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("El archivo de configuración no contiene la lista de clases.")

    label_to_index = metadata.get("label_to_index", {})
    if label_to_index:
        index_to_label = {int(index): str(label) for label, index in label_to_index.items()}
    else:
        index_to_label = {index: str(label) for index, label in enumerate(classes)}

    hog_config = metadata.get("hog", {})
    window = tuple(hog_config.get("window", [72, 96]))
    block = tuple(hog_config.get("block", [16, 16]))
    stride = tuple(hog_config.get("stride", [8, 8]))
    cell = tuple(hog_config.get("cell", [8, 8]))
    bins = int(hog_config.get("orientation_bins", 9))

    hog = cv2.HOGDescriptor(window, block, stride, cell, bins)
    svm = cv2.ml.SVM_load(str(model_path))
    if not svm.isTrained():
        raise ValueError("El archivo SVM no contiene un clasificador entrenado.")

    return svm, hog, window, index_to_label


def predict_crop(crop: np.ndarray, svm, hog, window, index_to_label) -> str:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, window, interpolation=cv2.INTER_AREA)
    descriptor = hog.compute(resized)
    if descriptor is None:
        raise ValueError("No fue posible extraer HOG de una posición.")

    sample = descriptor.reshape(1, -1).astype(np.float32)
    _, prediction = svm.predict(sample)
    predicted_index = int(round(float(prediction[0, 0])))
    if predicted_index not in index_to_label:
        raise ValueError(f"El modelo devolvió un índice de clase desconocido: {predicted_index}.")
    return index_to_label[predicted_index]


def annotate_prediction(image: np.ndarray, slot_number: int, class_id: str) -> None:
    index = slot_number - 1
    row = index // COLUMNS
    column = index % COLUMNS
    x0 = column * CELL_WIDTH
    y0 = row * CELL_HEIGHT
    x1 = x0 + CELL_WIDTH
    y1 = y0 + CELL_HEIGHT

    color = (40, 160, 75) if class_id != "empty" else (0, 145, 230)
    cv2.rectangle(image, (x0 + 3, y0 + 3), (x1 - 4, y1 - 4), color, 3)

    label = CLASS_NAMES_ES.get(class_id, class_id)
    text = f"{slot_number:02d} {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    left = x0 + 8
    top = y0 + 7
    right = min(x1 - 7, left + text_width + 10)
    bottom = top + text_height + baseline + 8
    cv2.rectangle(image, (left, top), (right, bottom), (250, 250, 245), -1)
    cv2.putText(
        image, text, (left + 5, bottom - baseline - 4),
        font, scale, (25, 35, 30), thickness, cv2.LINE_AA
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_image(image_path: Path, svm, hog, window, index_to_label,
                  output_dir: Path) -> tuple[list[dict], str]:
    """Realiza inferencia sin abrir ni consultar el JSON de referencia."""
    source = read_image(image_path)
    print(f"\nFotografía: {image_path.name} ({source.shape[1]} × {source.shape[0]} px)")
    print("Selecciona las cuatro esquinas exteriores de la cuadrícula 4×8.")
    print("Orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda.")

    corners = select_corners(source)
    rectified = rectify_shelf(source, corners)
    annotated = rectified.copy()

    rows = []
    crops_root = output_dir / "recortes_predichos"
    for class_id in sorted(index_to_label.values()):
        (crops_root / class_id).mkdir(parents=True, exist_ok=True)

    for slot_number in range(1, ROWS * COLUMNS + 1):
        crop = crop_slot(rectified, slot_number)
        predicted_class = predict_crop(crop, svm, hog, window, index_to_label)

        crop_path = crops_root / predicted_class / f"{image_path.stem}_slot_{slot_number:02d}.png"
        save_image(crop_path, crop)

        row_number = (slot_number - 1) // COLUMNS + 1
        column_number = (slot_number - 1) % COLUMNS + 1
        rows.append({
            "capture_id": image_path.stem,
            "source_image": image_path.name,
            "slot": slot_number,
            "row": row_number,
            "column": column_number,
            "predicted_class": predicted_class,
            "crop_path": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
        })
        annotate_prediction(annotated, slot_number, predicted_class)

    annotated_dir = output_dir / "anaqueles_anotados"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = annotated_dir / f"{image_path.stem}_predicho.png"
    save_image(annotated_path, annotated)

    print(f"  Posiciones clasificadas: {len(rows)}")
    print(f"  Anaquel anotado: {annotated_path}")
    return rows, str(annotated_path.relative_to(output_dir)).replace("\\", "/")


def compare_with_references(predictions: list[dict], references_dir: Path,
                            class_labels: list[str]) -> list[dict]:
    """Abre los JSON después de inferir y añade la etiqueta real a cada predicción."""
    predictions_by_capture: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        predictions_by_capture[row["capture_id"]].append(row)

    class_set = set(class_labels)
    compared = []
    for capture_id, capture_predictions in predictions_by_capture.items():
        reference_path = references_dir / f"{capture_id}.json"
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Falta el JSON de referencia de {capture_id}: {reference_path}"
            )

        labels_by_slot, _ = load_labels(reference_path)
        for row in capture_predictions:
            slot_number = int(row["slot"])
            slot = labels_by_slot[slot_number]
            real_class = str(slot.get("product_id")) if slot.get("product_id") else "empty"
            if real_class not in class_set:
                raise ValueError(
                    f"{reference_path.name}, posición {slot_number}: "
                    f"clase real no incluida en el modelo: {real_class}"
                )

            item = dict(row)
            item["real_class"] = real_class
            item["correct"] = real_class == row["predicted_class"]
            compared.append(item)

    return compared


def build_inventory_summary(predictions: list[dict], annotated_images: list[dict],
                            class_labels: list[str]) -> dict:
    """Resume las clases predichas por anaquel para reutilizarlas en la fase conversacional."""
    predictions_by_capture: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        predictions_by_capture[row["capture_id"]].append(row)

    annotated_by_capture = {
        item["capture_id"]: item["path"] for item in annotated_images
    }
    shelves = []
    for capture_id, rows in predictions_by_capture.items():
        counts = {
            label: sum(1 for row in rows if row["predicted_class"] == label)
            for label in class_labels
        }
        shelves.append({
            "capture_id": capture_id,
            "source_image": rows[0]["source_image"],
            "total_positions": len(rows),
            "occupied_predicted": len(rows) - counts.get("empty", 0),
            "empty_predicted": counts.get("empty", 0),
            "predicted_counts": counts,
            "annotated_image": annotated_by_capture.get(capture_id),
        })

    return {
        "model": "HOG + SVM",
        "classification_source": "JPG only; no reference JSON used for predictions",
        "note": (
            "Counts are visible shelf positions (facings), not warehouse stock. "
            "Each photograph is a separate shelf."
        ),
        "shelves": shelves,
    }


def calculate_metrics(rows: list[dict], class_labels: list[str]) -> dict:
    matrix = [[0 for _ in class_labels] for _ in class_labels]
    index = {label: number for number, label in enumerate(class_labels)}

    for row in rows:
        real_index = index[row["real_class"]]
        predicted_index = index[row["predicted_class"]]
        matrix[real_index][predicted_index] += 1

    per_class = []
    supported_f1 = []
    supported_precision = []
    supported_recall = []
    weighted_f1 = 0.0

    for i, label in enumerate(class_labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(matrix[row][i] for row in range(len(class_labels)))
        precision = tp / predicted_count if predicted_count else 0.0

        if support:
            recall = tp / support
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            supported_precision.append(precision)
            supported_recall.append(recall)
            supported_f1.append(f1)
            weighted_f1 += f1 * support
        else:
            recall = None
            f1 = None

        per_class.append({
            "class_id": label,
            "class_name": CLASS_NAMES_ES.get(label, label),
            "support": support,
            "predicted_count": predicted_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        })

    total = len(rows)
    correct = sum(matrix[i][i] for i in range(len(class_labels)))
    return {
        "total_positions": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(supported_precision) / len(supported_precision) if supported_precision else 0.0,
        "macro_recall": sum(supported_recall) / len(supported_recall) if supported_recall else 0.0,
        "macro_f1": sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
        "weighted_f1": weighted_f1 / total if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def generate_html_report(output_dir: Path, metrics: dict, class_labels: list[str],
                         annotated_images: list[dict], compared_rows: list[dict]) -> Path:
    errors = [row for row in compared_rows if not row["correct"]]
    shown_errors = errors[:32]

    metric_rows = []
    for item in metrics["per_class"]:
        metric_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(item['class_name'])}</strong></td>"
            f"<td>{item['support']}</td>"
            f"<td>{item['predicted_count']}</td>"
            f"<td>{pct(item['precision'])}</td>"
            f"<td>{pct(item['recall'])}</td>"
            f"<td>{pct(item['f1'])}</td>"
            "</tr>"
        )

    matrix = metrics["confusion_matrix"]
    maximum = max((max(row) for row in matrix), default=1) or 1
    matrix_rows = []
    for i, label in enumerate(class_labels):
        cells = []
        for j, value in enumerate(matrix[i]):
            intensity = value / maximum
            cells.append(
                f"<td class='heat' style='--heat:{0.10 + 0.68 * intensity:.3f}'>{value}</td>"
            )
        matrix_rows.append(
            f"<tr><th>{html.escape(CLASS_NAMES_ES.get(label, label))}</th>"
            + "".join(cells)
            + f"<td><strong>{sum(matrix[i])}</strong></td></tr>"
        )
    matrix_headers = "".join(
        f"<th>{html.escape(CLASS_NAMES_ES.get(label, label))}</th>"
        for label in class_labels
    )

    image_cards = []
    for item in annotated_images:
        relative = html.escape(item["path"], quote=True)
        caption = html.escape(item["capture_id"])
        image_cards.append(
            f"<article class='shelf-card'><h3>{caption}</h3>"
            f"<a href='{relative}' target='_blank'><img src='{relative}' alt='Predicciones de {caption}'></a>"
            "</article>"
        )

    error_cards = []
    for row in shown_errors:
        relative = html.escape(row["crop_path"], quote=True)
        real_name = html.escape(CLASS_NAMES_ES.get(row["real_class"], row["real_class"]))
        predicted_name = html.escape(CLASS_NAMES_ES.get(row["predicted_class"], row["predicted_class"]))
        capture = html.escape(row["capture_id"])
        slot = int(row["slot"])
        error_cards.append(
            f"<article class='error-card'><img src='{relative}' alt='Recorte mal clasificado'>"
            f"<div><strong>{real_name}</strong><span>→</span><strong>{predicted_name}</strong></div>"
            f"<small>{capture} · posición {slot}</small></article>"
        )

    if not error_cards:
        error_section = "<p>No se registraron errores en las posiciones evaluadas.</p>"
    else:
        suffix = (
            f"<p class='muted'>Se muestran {len(shown_errors)} de {len(errors)} errores. "
            "El archivo errores.csv contiene el detalle completo.</p>"
            if len(errors) > len(shown_errors) else ""
        )
        error_section = suffix + "<div class='error-grid'>" + "".join(error_cards) + "</div>"

    html_document = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluación del clasificador HOG + SVM</title>
<style>
:root {{ color-scheme: light; --ink:#19241f; --muted:#617168; --line:#dce5df; --paper:#f4f7f5; --green:#246b4b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 Segoe UI,Arial,sans-serif; }}
main {{ width:min(1180px,94vw); margin:30px auto 60px; }}
h1 {{ margin:4px 0 8px; font-size:32px; }}
h2 {{ font-size:21px; margin:0 0 12px; }}
h3 {{ font-size:15px; margin:0 0 8px; }}
p {{ margin:8px 0; }}
.muted {{ color:var(--muted); }}
header {{ margin-bottom:22px; }}
.eyebrow {{ color:var(--green); font-size:12px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.card, section {{ background:#fff; border:1px solid var(--line); border-radius:13px; }}
.card {{ padding:16px; }}
.label {{ color:var(--muted); font-size:12px; font-weight:700; }}
.value {{ font-size:29px; font-weight:800; margin-top:3px; }}
section {{ padding:20px; margin-top:16px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
thead th {{ background:#f4f8f5; }}
.matrix {{ min-width:800px; }}
.matrix th,.matrix td {{ text-align:center; }}
.matrix th:first-child {{ text-align:left; }}
.heat {{ background:rgba(36,107,75,var(--heat)); font-weight:700; }}
.shelf-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }}
.shelf-card,.error-card {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#fff; }}
.shelf-card {{ padding:10px; }}
.shelf-card img {{ width:100%; display:block; }}
.error-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(145px,1fr)); gap:10px; margin-top:12px; }}
.error-card img {{ width:100%; height:125px; object-fit:contain; background:#f1f4f2; padding:8px; }}
.error-card div {{ display:flex; gap:5px; flex-wrap:wrap; padding:8px 9px 2px; font-size:13px; }}
.error-card small {{ display:block; padding:0 9px 9px; color:var(--muted); }}
.note {{ background:#f1f6f2; border-left:4px solid #8ab49a; padding:12px 14px; border-radius:6px; }}
@media(max-width:700px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} section {{ padding:14px; }} }}
</style>
</head>
<body><main>
<header>
<div class="eyebrow">Tienda inteligente · Visión por computadora</div>
<h1>Evaluación del modelo HOG + SVM</h1>
<p class="muted">Comparación de las predicciones con las etiquetas reales de {len(annotated_images)} fotografías nuevas.</p>
</header>
<div class="cards">
<div class="card"><div class="label">Accuracy</div><div class="value">{pct(metrics['accuracy'])}</div><div class="muted">{metrics['correct']} de {metrics['total_positions']} posiciones correctas</div></div>
<div class="card"><div class="label">F1 macro</div><div class="value">{pct(metrics['macro_f1'])}</div><div class="muted">Promedio por categoría presente</div></div>
<div class="card"><div class="label">Errores</div><div class="value">{metrics['errors']}</div><div class="muted">Posiciones mal clasificadas</div></div>
<div class="card"><div class="label">Fotografías</div><div class="value">{len(annotated_images)}</div><div class="muted">Anaqueles independientes</div></div>
</div>
<section>
<h2>1. Rendimiento por categoría</h2>
<div class="table-wrap"><table>
<thead><tr><th>Categoría real</th><th>Ejemplos reales</th><th>Predicciones</th><th>Precisión</th><th>Recall</th><th>F1</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table></div>
</section>
<section>
<h2>2. Matriz de confusión</h2>
<p class="muted">Filas = clase real; columnas = clase predicha. La diagonal representa aciertos. Fuera de la diagonal aparecen las confusiones.</p>
<div class="table-wrap"><table class="matrix"><thead><tr><th>Real ↓ / Predicha →</th>{matrix_headers}<th>Total</th></tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div>
</section>
<section>
<h2>3. Errores del modelo</h2>
<p class="muted">Cada recorte muestra la categoría real y la que predijo el modelo.</p>
{error_section}
</section>
<section>
<h2>4. Anaqueles procesados</h2>
<div class="shelf-grid">{''.join(image_cards)}</div>
</section>
<section class="note">
<strong>Cómo interpretar esta evaluación:</strong>
Accuracy y las métricas se calculan sobre las posiciones de las fotografías, no sobre 160 fotografías independientes.
Las cinco capturas son una evaluación exploratoria con las mismas ilustraciones del generador; no demuestran reconocimiento de empaques comerciales reales.
</section>
</main></body></html>"""

    report_path = output_dir / "informe_evaluacion.html"
    report_path.write_text(html_document, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clasifica cinco anaqueles y evalúa el modelo con sus JSON de referencia."
    )
    parser.add_argument(
        "--entrada", type=Path,
        default=Path("dataset/iteraciones/evaluacion_01/entrada"),
        help="Carpeta con las fotografías JPG de los anaqueles.",
    )
    parser.add_argument(
        "--referencias", type=Path,
        default=Path("dataset/iteraciones/evaluacion_01/referencias"),
        help="Carpeta con los JSON de referencia, con el mismo nombre base que cada JPG.",
    )
    parser.add_argument(
        "--salida", type=Path,
        default=Path("salida_ml/iteraciones/evaluacion_01"),
        help="Carpeta donde se guardará el informe y los archivos de evaluación.",
    )
    parser.add_argument(
        "--modelo", type=Path, default=Path("salida_ml/modelo_hog_svm.yml"),
        help="Archivo del clasificador SVM entrenado.",
    )
    parser.add_argument(
        "--configuracion", type=Path, default=Path("salida_ml/modelo_hog_svm.json"),
        help="Configuración HOG y correspondencia de clases del modelo.",
    )
    parser.add_argument(
        "--sobrescribir", action="store_true",
        help="Elimina y vuelve a generar la carpeta de salida indicada.",
    )
    args = parser.parse_args()

    try:
        input_dir = resolve_from_project(args.entrada)
        references_dir = resolve_from_project(args.referencias)
        output_dir = resolve_from_project(args.salida)
        model_path = resolve_from_project(args.modelo)
        metadata_path = resolve_from_project(args.configuracion)

        if not input_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de fotografías: {input_dir}")
        if not references_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de referencias: {references_dir}")

        images = sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No se encontraron fotografías en {input_dir}")
        if len(images) != 5:
            print(f"AVISO: esta práctica contempla 5 fotografías; se encontraron {len(images)}.")

        # Solo se comprueba que existan los JSON. Su contenido no se lee en esta etapa.
        missing_references = [
            image.stem for image in images
            if not (references_dir / f"{image.stem}.json").is_file()
        ]
        if missing_references:
            raise FileNotFoundError(
                "Faltan JSON de referencia para: " + ", ".join(missing_references)
            )

        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.sobrescribir:
                raise FileExistsError(
                    f"La carpeta de salida ya contiene resultados: {output_dir}. "
                    "Usa --sobrescribir para regenerar esta evaluación."
                )
            if output_dir == BASE or BASE not in output_dir.parents:
                raise ValueError("La carpeta de salida debe estar dentro de cv-lab/.")

        if output_dir.exists() and args.sobrescribir:
            if output_dir == BASE or BASE not in output_dir.parents:
                raise ValueError("La carpeta de salida debe estar dentro de cv-lab/.")
            shutil.rmtree(output_dir)

        svm, hog, window, index_to_label = load_classifier(model_path, metadata_path)
        class_labels = sorted(set(index_to_label.values()))
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Fotografías encontradas: {len(images)}")
        print("FASE 1: clasificación. En esta fase no se leen los JSON de referencia.")
        all_predictions = []
        annotated_images = []

        for image_path in images:
            rows, annotated_path = process_image(
                image_path, svm, hog, window, index_to_label, output_dir
            )
            all_predictions.extend(rows)
            annotated_images.append({
                "capture_id": image_path.stem,
                "path": annotated_path,
            })

        inventory_summary = build_inventory_summary(
            all_predictions, annotated_images, class_labels
        )
        (output_dir / "inventario_por_foto.json").write_text(
            json.dumps(inventory_summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print("\nFASE 2: evaluación contra las etiquetas reales.")
        compared_rows = compare_with_references(
            all_predictions, references_dir, class_labels
        )
        metrics = calculate_metrics(compared_rows, class_labels)

        prediction_fields = [
            "capture_id", "source_image", "slot", "row", "column",
            "real_class", "predicted_class", "correct", "crop_path",
        ]
        write_csv(output_dir / "predicciones.csv", compared_rows, prediction_fields)

        write_csv(
            output_dir / "metricas_por_clase.csv",
            metrics["per_class"],
            ["class_id", "class_name", "support", "predicted_count",
             "precision", "recall", "f1"],
        )

        matrix_rows = []
        for label, values in zip(class_labels, metrics["confusion_matrix"]):
            matrix_rows.append({
                "real_class": label,
                **{f"pred_{pred_label}": value
                   for pred_label, value in zip(class_labels, values)},
            })
        write_csv(
            output_dir / "matriz_confusion.csv",
            matrix_rows,
            ["real_class", *[f"pred_{label}" for label in class_labels]],
        )

        errors = [row for row in compared_rows if not row["correct"]]
        write_csv(
            output_dir / "errores.csv",
            errors,
            prediction_fields,
        )

        summary = {
            "photographs": len(images),
            "positions": metrics["total_positions"],
            "correct": metrics["correct"],
            "errors": metrics["errors"],
            "accuracy": round(metrics["accuracy"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "per_class": metrics["per_class"],
            "confusion_matrix_labels": class_labels,
            "confusion_matrix": metrics["confusion_matrix"],
            "reference_usage": "JSON abierto únicamente después de completar todas las predicciones",
            "note": "Evaluación exploratoria por posición; las posiciones de una misma fotografía no son observaciones independientes.",
        }
        (output_dir / "evaluacion.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        report_path = generate_html_report(
            output_dir, metrics, class_labels, annotated_images, compared_rows
        )

        print("\nEvaluación finalizada.")
        print(f"  Fotografías: {len(images)}")
        print(f"  Posiciones evaluadas: {metrics['total_positions']}")
        print(f"  Accuracy: {pct(metrics['accuracy'])}")
        print(f"  F1 macro: {pct(metrics['macro_f1'])}")
        print(f"  Errores: {metrics['errors']}")
        print(f"  Informe para revisar: {report_path}")
        print("\nLos JSON se utilizaron únicamente después de la clasificación, como referencia de evaluación.")
        return 0

    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError,
            json.JSONDecodeError, cv2.error) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
