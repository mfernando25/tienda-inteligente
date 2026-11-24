#!/usr/bin/env python3
"""Clasifica comentarios con un modelo Transformer preentrenado en español."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pysentimiento import create_analyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
SENTIMENTS = ("positivo", "negativo", "neutral")
MODEL_LABELS = {"POS": "positivo", "NEG": "negativo", "NEU": "neutral"}
PRODUCT_NAMES = {
    "canned": "Conservas",
    "cereal": "Cereal",
    "cookies": "Galletas",
    "detergent": "Detergente",
    "juice": "Jugo",
    "milk": "Leche",
    "soda": "Gaseosa",
    "water": "Agua",
}


def load_comments(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")

    required = {
        "comentario_id", "product_id", "producto",
        "comentario", "sentimiento_referencia",
    }
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Faltan columnas obligatorias: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        raise ValueError("El dataset no contiene comentarios.")

    seen_ids = set()
    for row in rows:
        comment_id = row["comentario_id"].strip()
        product_id = row["product_id"].strip()
        reference = row["sentimiento_referencia"].strip().lower()
        if not comment_id or comment_id in seen_ids:
            raise ValueError(f"Identificador vacío o repetido: {comment_id!r}")
        seen_ids.add(comment_id)
        if product_id not in PRODUCT_NAMES:
            raise ValueError(f"Categoría no reconocida en {comment_id}: {product_id}")
        if reference not in SENTIMENTS:
            raise ValueError(f"Sentimiento de referencia no válido en {comment_id}: {reference}")
        if not row["comentario"].strip():
            raise ValueError(f"Comentario vacío: {comment_id}")
    return rows


def label_from_model(output) -> str:
    raw_label = getattr(output, "output", None)
    if raw_label is None:
        raise ValueError("El modelo no devolvió una etiqueta de sentimiento.")
    key = str(getattr(raw_label, "value", raw_label)).upper()
    if key not in MODEL_LABELS:
        raise ValueError(f"Etiqueta inesperada devuelta por el modelo: {key}")
    return MODEL_LABELS[key]


def build_evaluation(rows: list[dict]) -> dict:
    y_true = [row["sentimiento_referencia"] for row in rows]
    y_pred = [row["sentimiento_predicho"] for row in rows]
    report = classification_report(
        y_true, y_pred, labels=list(SENTIMENTS),
        target_names=list(SENTIMENTS), output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(SENTIMENTS))
    return {
        "total_comentarios": len(rows),
        "aciertos": sum(real == predicted for real, predicted in zip(y_true, y_pred)),
        "errores": sum(real != predicted for real, predicted in zip(y_true, y_pred)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(report["macro avg"]["f1-score"]), 4),
        "matriz_confusion": {
            real: {pred: int(matrix[i][j]) for j, pred in enumerate(SENTIMENTS)}
            for i, real in enumerate(SENTIMENTS)
        },
        "metricas_por_sentimiento": [
            {
                "sentimiento": label,
                "soporte": int(report[label]["support"]),
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in SENTIMENTS
        ],
    }


def build_product_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["product_id"]].append(row)

    products = []
    for product_id, product_rows in grouped.items():
        counts = Counter(row["sentimiento_predicho"] for row in product_rows)
        comments = [
            {
                "comentario_id": row["comentario_id"],
                "comentario": row["comentario"],
                "sentimiento": row["sentimiento_predicho"],
                "confianza": row["confianza"],
            }
            for row in product_rows
        ]
        products.append({
            "product_id": product_id,
            "producto": PRODUCT_NAMES[product_id],
            "total_comentarios": len(product_rows),
            "sentimientos": {label: counts.get(label, 0) for label in SENTIMENTS},
            "comentarios_negativos": [
                item for item in comments if item["sentimiento"] == "negativo"
            ],
            "comentarios": comments,
        })

    products.sort(key=lambda item: item["producto"])
    return {
        "fuente": "nlp-lab/dataset/comentarios_clientes.csv",
        "metodo": "modelo Transformer preentrenado (PySentimiento, español)",
        "total_comentarios": len(rows),
        "productos": products,
        "nota": (
            "Modelo preentrenado para análisis de sentimiento en español. "
            "El modelo de referencia se entrenó principalmente con publicaciones de redes sociales; "
            "su rendimiento puede variar en opiniones comerciales. Los comentarios del dataset son simulados."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clasifica comentarios en español con un Transformer preentrenado."
    )
    parser.add_argument("--entrada", type=Path, default=Path("dataset/comentarios_clientes.csv"))
    parser.add_argument("--salida", type=Path, default=Path("salida"))
    args = parser.parse_args()

    try:
        csv_path = args.entrada if args.entrada.is_absolute() else BASE / args.entrada
        output_dir = args.salida if args.salida.is_absolute() else BASE / args.salida
        source_rows = load_comments(csv_path)

        print("Cargando el modelo preentrenado de análisis de sentimientos en español...")
        analyzer = create_analyzer(task="sentiment", lang="es")
        model_outputs = analyzer.predict([row["comentario"] for row in source_rows])
        if not isinstance(model_outputs, (list, tuple)):
            model_outputs = [model_outputs]
        if len(model_outputs) != len(source_rows):
            raise ValueError("El modelo devolvió una cantidad inesperada de predicciones.")

        analysed = []
        for row, prediction in zip(source_rows, model_outputs):
            probabilities = {
                str(label).upper(): float(score)
                for label, score in prediction.probas.items()
            }
            predicted = label_from_model(prediction)
            analysed.append({
                "comentario_id": row["comentario_id"],
                "product_id": row["product_id"],
                "producto": row["producto"],
                "comentario": row["comentario"],
                "sentimiento_referencia": row["sentimiento_referencia"].strip().lower(),
                "sentimiento_predicho": predicted,
                "acierto": predicted == row["sentimiento_referencia"].strip().lower(),
                "confianza": round(max(probabilities.values()), 4),
                "probabilidad_positivo": round(probabilities.get("POS", 0.0), 4),
                "probabilidad_negativo": round(probabilities.get("NEG", 0.0), 4),
                "probabilidad_neutral": round(probabilities.get("NEU", 0.0), 4),
            })

        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "comentarios_analizados.csv", analysed)
        evaluation = build_evaluation(analysed)
        (output_dir / "evaluacion_sentimientos.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = build_product_summary(analysed)
        (output_dir / "resumen_sentimientos.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("Análisis finalizado.")
        print(f"  Comentarios: {len(analysed)}")
        print(f"  Accuracy frente a las etiquetas de referencia: {evaluation['accuracy'] * 100:.1f}%")
        print(f"  F1 macro: {evaluation['f1_macro'] * 100:.1f}%")
        print(f"  Resultados: {output_dir.resolve()}")
        print("  Archivo que utilizará el chatbot: resumen_sentimientos.json")
        print("Revisa los desacuerdos: el modelo es general y el dataset contiene opiniones simuladas.")
        return 0

    except ImportError as exc:
        parser.exit(1, f"ERROR: falta una dependencia. Ejecuta: python -m pip install -r nlp-lab/requirements.txt\nDetalle: {exc}\n")
    except (FileNotFoundError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
