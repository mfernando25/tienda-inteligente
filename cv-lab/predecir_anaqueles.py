#!/usr/bin/env python3
"""Clasifica por lotes las 32 posiciones de cada fotografía de anaquel."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
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
from preparar_dataset import crop_slot

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
    """Resolve a relative path from cv-lab, even if called from elsewhere."""
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


def predict_crop(crop: np.ndarray, svm, hog, window: tuple[int, int],
                 index_to_label: dict[int, str]) -> str:
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
    cv2.putText(image, text, (left + 5, bottom - baseline - 4),
                font, scale, (25, 35, 30), thickness, cv2.LINE_AA)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_image(image_path: Path, svm, hog, window, index_to_label,
                  output_dir: Path) -> tuple[list[dict], dict]:
    source = read_image(image_path)
    print(f"\nFotografía: {image_path.name} ({source.shape[1]} × {source.shape[0]} px)")
    print("Selecciona las cuatro esquinas exteriores de la cuadrícula 4×8.")
    print("Orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda.")

    corners = select_corners(source)
    rectified = rectify_shelf(source, corners)
    annotated = rectified.copy()

    rows = []
    counts: Counter = Counter()
    crops_root = output_dir / "recortes_predichos"

    for slot_number in range(1, ROWS * COLUMNS + 1):
        crop = crop_slot(rectified, slot_number)
        predicted_class = predict_crop(crop, svm, hog, window, index_to_label)

        class_dir = crops_root / predicted_class
        class_dir.mkdir(parents=True, exist_ok=True)
        crop_path = class_dir / f"{image_path.stem}_slot_{slot_number:02d}.png"
        save_image(crop_path, crop)

        row_number = (slot_number - 1) // COLUMNS + 1
        column_number = (slot_number - 1) % COLUMNS + 1
        rows.append({
            "capture_id": image_path.stem,
            "source_image": str(image_path),
            "slot": slot_number,
            "row": row_number,
            "column": column_number,
            "predicted_class": predicted_class,
            "crop_path": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
        })
        counts[predicted_class] += 1
        annotate_prediction(annotated, slot_number, predicted_class)

    annotated_dir = output_dir / "anaqueles_anotados"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = annotated_dir / f"{image_path.stem}_predicho.png"
    save_image(annotated_path, annotated)

    inventory = {
        "capture_id": image_path.stem,
        "source_image": image_path.name,
        "total_positions": ROWS * COLUMNS,
        "occupied_predicted": ROWS * COLUMNS - counts.get("empty", 0),
        "empty_predicted": counts.get("empty", 0),
        "predicted_counts": {label: counts.get(label, 0) for label in sorted(index_to_label.values())},
        "annotated_image": str(annotated_path.relative_to(output_dir)).replace("\\", "/"),
    }
    print(f"  Clasificaciones: {len(rows)}")
    print(f"  Posiciones vacías predichas: {inventory['empty_predicted']}")
    print(f"  Anaquel anotado: {annotated_path}")
    return rows, inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aplica HOG + SVM a un lote de fotografías completas de anaqueles."
    )
    parser.add_argument(
        "--entrada", type=Path,
        default=Path("dataset/iteraciones/iteracion_01/entrada"),
        help="Carpeta con fotografías JPG/PNG de anaqueles completos.",
    )
    parser.add_argument(
        "--salida", type=Path,
        default=Path("salida_ml/iteraciones/iteracion_01"),
        help="Carpeta donde se guardarán los resultados de esta iteración.",
    )
    parser.add_argument(
        "--modelo", type=Path, default=Path("salida_ml/modelo_hog_svm.yml"),
        help="Archivo del clasificador SVM ya entrenado.",
    )
    parser.add_argument(
        "--configuracion", type=Path, default=Path("salida_ml/modelo_hog_svm.json"),
        help="Configuración HOG y correspondencia de clases del modelo.",
    )
    parser.add_argument(
        "--sobrescribir", action="store_true",
        help="Reemplaza los resultados existentes de esta iteración.",
    )
    args = parser.parse_args()

    try:
        input_dir = resolve_from_project(args.entrada)
        output_dir = resolve_from_project(args.salida)
        model_path = resolve_from_project(args.modelo)
        metadata_path = resolve_from_project(args.configuracion)

        if not input_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

        images = sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No se encontraron fotografías en {input_dir}")

        if output_dir.exists() and any(output_dir.iterdir()) and not args.sobrescribir:
            raise FileExistsError(
                f"La carpeta de salida ya contiene resultados: {output_dir}. "
                "Usa --sobrescribir para reemplazar esta iteración."
            )

        print(f"Fotografías encontradas: {len(images)}")
        if len(images) != 15:
            print("AVISO: la planificación de esta iteración contempla 15 fotografías.")

        svm, hog, window, index_to_label = load_classifier(model_path, metadata_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir = output_dir / "anaqueles_anotados"
        crops_dir = output_dir / "recortes_predichos"

        if args.sobrescribir:
            for folder in (annotated_dir, crops_dir):
                if folder.exists():
                    shutil.rmtree(folder)

        all_predictions = []
        inventories = []

        for image_path in images:
            prediction_rows, inventory = process_image(
                image_path, svm, hog, window, index_to_label, output_dir
            )
            all_predictions.extend(prediction_rows)
            inventories.append(inventory)
            print(
                "  Inventario estimado: "
                + ", ".join(
                    f"{CLASS_NAMES_ES.get(class_id, class_id)}={count}"
                    for class_id, count in inventory["predicted_counts"].items()
                )
            )

        prediction_fields = [
            "capture_id", "source_image", "slot", "row", "column",
            "predicted_class", "crop_path",
        ]
        write_csv(output_dir / "predicciones.csv", all_predictions, prediction_fields)

        inventory_fields = [
            "capture_id", "source_image", "total_positions",
            "occupied_predicted", "empty_predicted",
            *[f"pred_{label}" for label in sorted(index_to_label.values())],
        ]
        inventory_rows = []
        for inventory in inventories:
            row = {
                "capture_id": inventory["capture_id"],
                "source_image": inventory["source_image"],
                "total_positions": inventory["total_positions"],
                "occupied_predicted": inventory["occupied_predicted"],
                "empty_predicted": inventory["empty_predicted"],
            }
            row.update({
                f"pred_{label}": count
                for label, count in inventory["predicted_counts"].items()
            })
            inventory_rows.append(row)
        write_csv(output_dir / "inventario_por_foto.csv", inventory_rows, inventory_fields)

        summary = {
            "iteration": output_dir.name,
            "model": str(model_path),
            "number_of_photographs": len(images),
            "classification_source": "photograph only; no shelf JSON is loaded",
            "note": (
                "Cada fotografía es un anaquel independiente. "
                "No sumar sus conteos como si fueran un único inventario físico."
            ),
            "shelves": inventories,
        }
        (output_dir / "inventario_por_foto.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("\nIteración finalizada.")
        print(f"  Fotografías procesadas: {len(inventories)}")
        print(f"  Posiciones clasificadas: {len(all_predictions)}")
        print(f"  Recortes por categoría: {crops_dir}")
        print(f"  Predicciones: {output_dir / 'predicciones.csv'}")
        print(f"  Inventario por fotografía: {output_dir / 'inventario_por_foto.csv'}")
        print(f"  Resumen JSON: {output_dir / 'inventario_por_foto.json'}")
        print("\nEl JSON del generador no se utilizó para realizar las predicciones.")
        return 0

    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
