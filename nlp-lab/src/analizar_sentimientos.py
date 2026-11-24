#!/usr/bin/env python3
"""Analiza opiniones en español con un enfoque léxico basado en reglas.

No entrena un modelo ni llama servicios externos. Compara la predicción con
sentimiento_referencia para que los estudiantes observen aciertos y límites.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SENTIMENTS = ("positivo", "negativo", "neutral")
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

# Diccionarios iniciales para esta actividad. Se pueden ampliar y discutir.
POSITIVE_WORDS = {
    "agradable", "agradables", "buena", "buenas", "bueno", "buenos", "buen",
    "bien", "cremosa", "cremoso", "crocante", "crocantes", "correctamente",
    "delicioso", "deliciosa", "facil", "fresca", "fresco", "frescas", "frescos",
    "gusta", "gusto", "ligera", "ligero", "natural", "naturales", "refrescante",
    "rendidor", "rendidoras", "rendidores", "rendidora", "rinde", "suave",
    "suaves", "suficiente", "suficientes", "sellada", "sellado", "intacto",
    "intacta", "alcanza", "quita", "elimina", "rapido", "rapida", "ligero",
}

NEGATIVE_WORDS = {
    "agria", "agrio", "aguada", "aguado", "artificial", "artificiales",
    "abierta", "abierto", "caliente", "danada", "danado", "desagradable",
    "desagradables", "derramo", "empalagada", "empalagado", "empalagosa",
    "empalagoso", "extrana", "extrano", "golpeada", "golpeado", "hinchada",
    "hinchado", "humeda", "humedo", "mal", "mala", "malo", "olor", "perdio",
    "pocas", "roto", "rota", "salado", "salada", "plano", "residuos",
    "vencida", "vencido", "jarabe", "demasiado", "pésimo", "pesimo",
}

NEGATORS = {"no", "nunca", "jamas", "tampoco"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase and remove diacritics for consistent dictionary matching."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def classify_sentiment(tokens: list[str]) -> tuple[str, int, int]:
    """Count lexicon matches, inverting a match when preceded by a negator."""
    positive_score = 0
    negative_score = 0

    for index, token in enumerate(tokens):
        polarity = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if polarity == 0:
            continue

        previous_tokens = tokens[max(0, index - 3):index]
        if any(previous in NEGATORS for previous in previous_tokens):
            polarity *= -1

        if polarity > 0:
            positive_score += 1
        else:
            negative_score += 1

    if positive_score > negative_score:
        predicted = "positivo"
    elif negative_score > positive_score:
        predicted = "negativo"
    else:
        predicted = "neutral"

    return predicted, positive_score, negative_score


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


def calculate_evaluation(rows: list[dict]) -> dict:
    matrix = {real: {predicted: 0 for predicted in SENTIMENTS} for real in SENTIMENTS}
    for row in rows:
        matrix[row["sentimiento_referencia"]][row["sentimiento_predicho"]] += 1

    per_class = []
    f1_values = []
    total_correct = 0

    for sentiment in SENTIMENTS:
        true_positive = matrix[sentiment][sentiment]
        support = sum(matrix[sentiment].values())
        predicted_total = sum(matrix[real][sentiment] for real in SENTIMENTS)
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        if support:
            f1_values.append(f1)
        total_correct += true_positive
        per_class.append({
            "sentimiento": sentiment,
            "soporte": support,
            "predicciones": predicted_total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    total = len(rows)
    return {
        "total_comentarios": total,
        "aciertos": total_correct,
        "errores": total - total_correct,
        "accuracy": round(total_correct / total, 4) if total else 0.0,
        "f1_macro": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "matriz_confusion": matrix,
        "metricas_por_sentimiento": per_class,
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
            }
            for row in product_rows
        ]
        products.append({
            "product_id": product_id,
            "producto": PRODUCT_NAMES[product_id],
            "total_comentarios": len(product_rows),
            "sentimientos": {sentiment: counts.get(sentiment, 0) for sentiment in SENTIMENTS},
            "comentarios_negativos": [
                item for item in comments if item["sentimiento"] == "negativo"
            ],
            "comentarios": comments,
        })

    products.sort(key=lambda item: item["producto"])
    return {
        "fuente": "nlp-lab/dataset/comentarios_clientes.csv",
        "metodo": "clasificacion lexica basada en reglas",
        "total_comentarios": len(rows),
        "productos": products,
        "nota": (
            "Los sentimientos son predicciones de un diccionario y reglas simples. "
            "No equivalen a una comprensión completa del contexto o la ironía."
        ),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Limpia, tokeniza y clasifica comentarios de clientes en español."
    )
    parser.add_argument(
        "--entrada", type=Path,
        default=Path("dataset/comentarios_clientes.csv"),
        help="CSV de opiniones con etiquetas de referencia.",
    )
    parser.add_argument(
        "--salida", type=Path,
        default=Path("salida"),
        help="Carpeta de resultados del análisis.",
    )
    args = parser.parse_args()

    try:
        csv_path = args.entrada if args.entrada.is_absolute() else BASE / args.entrada
        output_dir = args.salida if args.salida.is_absolute() else BASE / args.salida
        source_rows = load_comments(csv_path)

        analysed = []
        for row in source_rows:
            tokens = tokenize(row["comentario"])
            predicted, positive_score, negative_score = classify_sentiment(tokens)
            analysed.append({
                "comentario_id": row["comentario_id"],
                "product_id": row["product_id"],
                "producto": row["producto"],
                "comentario": row["comentario"],
                "texto_limpio": " ".join(tokens),
                "tokens": " | ".join(tokens),
                "sentimiento_referencia": row["sentimiento_referencia"].strip().lower(),
                "sentimiento_predicho": predicted,
                "acierto": predicted == row["sentimiento_referencia"].strip().lower(),
                "puntaje_positivo": positive_score,
                "puntaje_negativo": negative_score,
            })

        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            output_dir / "comentarios_analizados.csv",
            analysed,
            list(analysed[0].keys()),
        )

        evaluation = calculate_evaluation(analysed)
        (output_dir / "evaluacion_sentimientos.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = build_product_summary(analysed)
        (output_dir / "resumen_sentimientos.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("Análisis de opiniones finalizado.")
        print(f"  Comentarios procesados: {len(analysed)}")
        print(f"  Accuracy de referencia: {evaluation['accuracy'] * 100:.1f}%")
        print(f"  F1 macro: {evaluation['f1_macro'] * 100:.1f}%")
        print(f"  Resultados: {output_dir.resolve()}")
        print("  Archivo para el sistema conversacional: resumen_sentimientos.json")
        print("\nNota: el método es léxico y basado en reglas; revisa los errores antes de interpretarlo.")
        return 0

    except (FileNotFoundError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
