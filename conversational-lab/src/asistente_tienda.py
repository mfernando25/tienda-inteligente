#!/usr/bin/env python3
"""Chatbot que aprende a clasificar intenciones y consulta datos de la tienda."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer

REPO_ROOT = Path(__file__).resolve().parents[2]
INTENT_DATASET = Path(__file__).resolve().parents[1] / "dataset" / "intenciones.csv"
MIN_INTENT_SCORE = 0.20
MIN_INTENT_MARGIN = 0.05
SPANISH_STEMMER = SnowballStemmer("spanish")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

PRODUCTS = {
    "canned": {"name": "Conservas", "aliases": ("conserva", "conservas", "lata", "latas")},
    "cereal": {"name": "Cereal", "aliases": ("cereal", "cereales")},
    "cookies": {"name": "Galletas", "aliases": ("galleta", "galletas")},
    "detergent": {"name": "Detergente", "aliases": ("detergente", "detergentes", "jabon", "lavado")},
    "juice": {"name": "Jugo", "aliases": ("jugo", "jugos", "zumo")},
    "milk": {"name": "Leche", "aliases": ("leche",)},
    "soda": {"name": "Gaseosa", "aliases": ("gaseosa", "gaseosas", "refresco")},
    "water": {"name": "Agua", "aliases": ("agua", "aguas")},
}


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def stem_spanish_tokens(text: str) -> list[str]:
    """Reduce flexiones comunes del español para reconocer singular/plural y verbos."""
    normalized = normalize(text)
    return [SPANISH_STEMMER.stem(token) for token in TOKEN_PATTERN.findall(normalized)]


def load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"El archivo {description} no contiene JSON válido: {path}") from exc


def load_sources(inventory_path: Path, opinions_path: Path) -> tuple[list[dict], dict[str, dict]]:
    inventory = load_json(inventory_path, "inventario visual")
    opinions = load_json(opinions_path, "resumen de sentimientos")
    shelves = inventory.get("shelves")
    products = opinions.get("productos")
    if not isinstance(shelves, list) or not shelves:
        raise ValueError("El JSON de inventario no contiene una lista 'shelves'.")
    if not isinstance(products, list) or not products:
        raise ValueError("El JSON de opiniones no contiene una lista 'productos'.")
    for shelf in shelves:
        if not isinstance(shelf.get("predicted_counts"), dict):
            raise ValueError("Una entrada del inventario no contiene 'predicted_counts'.")
    opinion_by_id = {
        item.get("product_id"): item for item in products if item.get("product_id")
    }
    return shelves, opinion_by_id


def train_intent_classifier(dataset_path: Path):
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset de intenciones: {dataset_path}")
    with dataset_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 20:
        raise ValueError("El dataset de intenciones necesita al menos 20 ejemplos etiquetados.")
    if not all(row.get("consulta", "").strip() and row.get("intencion", "").strip() for row in rows):
        raise ValueError("Cada fila de intenciones debe incluir consulta e intencion.")

    questions = [row["consulta"].strip() for row in rows]
    labels = [row["intencion"].strip() for row in rows]
    if len(set(labels)) < 3:
        raise ValueError("El dataset debe contener al menos tres intenciones distintas.")

    # Reserva una parte para medir la generalización; después vuelve a entrenar
    # el clasificador con todos los ejemplos disponibles para usarlo en el chatbot.
    train_questions, test_questions, train_labels, test_labels = train_test_split(
        questions, labels, test_size=0.25, random_state=42, stratify=labels,
    )

    def create_pipeline():
        return make_pipeline(
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=5000,
            ),
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        )

    evaluation_model = create_pipeline()
    evaluation_model.fit(train_questions, train_labels)
    test_predictions = evaluation_model.predict(test_questions)
    validation_accuracy = float(accuracy_score(test_labels, test_predictions))
    validation_f1 = float(f1_score(test_labels, test_predictions, average="macro", zero_division=0))

    classifier = create_pipeline()
    classifier.fit(questions, labels)
    return classifier, len(rows), len(set(labels)), validation_accuracy, validation_f1


def find_product(query: str) -> str | None:
    normalized = normalize(query)
    matches = []
    for product_id, data in PRODUCTS.items():
        for alias in data["aliases"]:
            if re.search(rf"\b{re.escape(normalize(alias))}\b", normalized):
                matches.append((len(alias), product_id))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def aggregate_inventory(shelves: list[dict]) -> dict[str, dict]:
    summary = {
        product_id: {"positions": 0, "present_shelves": 0}
        for product_id in PRODUCTS
    }
    for shelf in shelves:
        counts = shelf.get("predicted_counts", {})
        for product_id in PRODUCTS:
            count = int(counts.get(product_id, 0) or 0)
            summary[product_id]["positions"] += count
            if count > 0:
                summary[product_id]["present_shelves"] += 1
    for data in summary.values():
        data["absent_shelves"] = len(shelves) - data["present_shelves"]
    return summary


def format_products(items: list[tuple[str, int]], suffix: str) -> str:
    return "; ".join(
        f"{PRODUCTS[product_id]['name']} ({count} {suffix})"
        for product_id, count in items
    )


def answer_least_present(shelves: list[dict]) -> str:
    summary = aggregate_inventory(shelves)
    minimum = min(item["positions"] for item in summary.values())
    leaders = [
        (product_id, item["positions"])
        for product_id, item in summary.items()
        if item["positions"] == minimum
    ]
    return (
        "Productos con menor presencia: "
        + format_products(leaders, "posiciones visibles")
        + f". Comparación basada en {len(shelves)} fotografías."
    )


def answer_most_present(shelves: list[dict]) -> str:
    summary = aggregate_inventory(shelves)
    maximum = max(item["positions"] for item in summary.values())
    leaders = [
        (product_id, item["positions"])
        for product_id, item in summary.items()
        if item["positions"] == maximum
    ]
    occupied = sum(item["positions"] for item in summary.values())
    percentage = maximum * len(leaders) / occupied * 100 if occupied else 0.0
    return (
        "Productos con mayor presencia: "
        + format_products(leaders, "posiciones visibles")
        + f". En conjunto representan {percentage:.1f}% de las posiciones ocupadas."
    )


def answer_absences(shelves: list[dict]) -> str:
    summary = aggregate_inventory(shelves)
    maximum = max(item["absent_shelves"] for item in summary.values())
    if maximum == 0:
        return "Todos los productos aparecen en las cinco fotografías analizadas."
    leaders = [
        (product_id, item["absent_shelves"])
        for product_id, item in summary.items()
        if item["absent_shelves"] == maximum
    ]
    details = "; ".join(
        f"{PRODUCTS[product_id]['name']} (ausente en {count} de {len(shelves)} fotografías)"
        for product_id, count in leaders
    )
    return (
        "Productos con más ausencias en la muestra: " + details
        + ". Ausencia en la exhibición no demuestra falta de stock en almacén."
    )


def answer_occupancy(shelves: list[dict], query: str) -> str:
    want_empty = any(term in normalize(query) for term in (
        "vacio", "vacios", "libre", "libres", "disponible", "disponibles",
    ))
    ranked = sorted(
        shelves,
        key=lambda shelf: (
            shelf.get("empty_predicted", 0) / shelf.get("total_positions", 1)
            if shelf.get("total_positions", 0) else 0.0
        ),
        reverse=want_empty,
    )
    selected = ranked[:2]
    label = "más espacios libres" if want_empty else "mayor ocupación"
    details = []
    for item in selected:
        total = item.get("total_positions", 0)
        occupied = item.get("occupied_predicted", 0)
        empty = item.get("empty_predicted", 0)
        details.append(
            f"{item.get('capture_id', 'sin identificar')}: "
            f"{occupied}/{total} ocupadas y {empty} vacías"
        )
    return f"Anaqueles con {label}: " + "; ".join(details) + "."


def answer_product_inventory(shelves: list[dict], product_id: str) -> str:
    summary = aggregate_inventory(shelves)[product_id]
    return (
        f"{PRODUCTS[product_id]['name']}: {summary['positions']} posiciones visibles "
        f"en total, presente en {summary['present_shelves']} de {len(shelves)} fotografías."
    )


def answer_product_opinions(opinion_by_id: dict[str, dict], product_id: str,
                            negative_only: bool) -> str:
    item = opinion_by_id.get(product_id)
    if not item:
        return f"No hay comentarios disponibles para {PRODUCTS[product_id]['name']}."
    counts = item.get("sentimientos", {})
    name = item.get("producto", PRODUCTS[product_id]["name"])
    response = (
        f"{name}: {counts.get('positivo', 0)} positivas, "
        f"{counts.get('negativo', 0)} negativas y "
        f"{counts.get('neutral', 0)} neutrales "
        f"({item.get('total_comentarios', 0)} comentarios analizados)."
    )
    if negative_only:
        examples = item.get("comentarios_negativos", [])[:3]
        if examples:
            response += "\nComentarios negativos clasificados:\n" + "\n".join(
                f"- «{entry.get('comentario', '')}»" for entry in examples
            )
        else:
            response += "\nNo se clasificaron comentarios negativos."
    return response


def answer_most_negative(opinion_by_id: dict[str, dict]) -> str:
    ranked = sorted(
        opinion_by_id.values(),
        key=lambda item: item.get("sentimientos", {}).get("negativo", 0),
        reverse=True,
    )
    if not ranked:
        return "No hay opiniones disponibles."
    maximum = ranked[0].get("sentimientos", {}).get("negativo", 0)
    leaders = [
        item.get("producto", item.get("product_id"))
        for item in ranked
        if item.get("sentimientos", {}).get("negativo", 0) == maximum
    ]
    return (
        "Productos con más comentarios negativos clasificados: "
        + ", ".join(leaders)
        + f" ({maximum} por categoría)."
    )


def answer(query: str, intent: str, shelves: list[dict],
           opinion_by_id: dict[str, dict]) -> str:
    product_id = find_product(query)

    if intent == "MENOR_PRESENCIA":
        return answer_least_present(shelves)
    if intent == "MAYOR_PRESENCIA":
        return answer_most_present(shelves)
    if intent == "PRODUCTOS_AUSENTES":
        return answer_absences(shelves)
    if intent == "OCUPACION_ANAQUELES":
        return answer_occupancy(shelves, query)
    if intent == "INVENTARIO_PRODUCTO":
        if not product_id:
            return "¿De qué producto necesitas consultar las posiciones visibles?"
        return answer_product_inventory(shelves, product_id)
    if intent == "OPINIONES_PRODUCTO":
        if not product_id:
            return "¿Sobre qué producto quieres conocer las opiniones de los clientes?"
        return answer_product_opinions(opinion_by_id, product_id, negative_only=False)
    if intent == "QUEJAS_PRODUCTO":
        if not product_id:
            return "¿De qué producto quieres revisar los comentarios negativos?"
        return answer_product_opinions(opinion_by_id, product_id, negative_only=True)
    if intent == "MAS_QUEJAS":
        return answer_most_negative(opinion_by_id)
    if intent == "CONSULTA_TEMPORAL":
        return (
            "No puedo determinar cuánto tiempo lleva faltando un producto. "
            "El proyecto contiene fotografías de anaqueles distintos, pero no incluye "
            "historial de fechas, disponibilidad ni reposiciones."
        )
    return "No reconocí la intención. Reformula la pregunta dentro del contexto de inventario u opiniones de clientes."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chatbot con clasificación de intenciones mediante aprendizaje automático."
    )
    parser.add_argument(
        "--inventario", type=Path,
        default=Path("cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json"),
    )
    parser.add_argument(
        "--opiniones", type=Path,
        default=Path("nlp-lab/salida/resumen_sentimientos.json"),
    )
    parser.add_argument(
        "--intenciones", type=Path, default=INTENT_DATASET,
    )
    args = parser.parse_args()

    inventory_path = args.inventario if args.inventario.is_absolute() else REPO_ROOT / args.inventario
    opinions_path = args.opiniones if args.opiniones.is_absolute() else REPO_ROOT / args.opiniones
    intent_path = args.intenciones if args.intenciones.is_absolute() else REPO_ROOT / args.intenciones

    try:
        shelves, opinion_by_id = load_sources(inventory_path, opinions_path)
        classifier, example_count, intent_count, intent_accuracy, intent_f1 = train_intent_classifier(intent_path)
    except (FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print("TIENDA INTELIGENTE – Chatbot de consulta")
    print(f"Clasificador entrenado con {example_count} preguntas y {intent_count} intenciones.")
    print(f"Validación (25% reservado): accuracy {intent_accuracy * 100:.1f}% | F1 macro {intent_f1 * 100:.1f}%.")
    print("El modelo usado en la conversación se vuelve a entrenar con todos los ejemplos.")
    print("Fuentes: inventario visual y opiniones analizadas.")
    print("Escribe 'ayuda' para ejemplos o 'salir' para terminar.\n")

    while True:
        try:
            query = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAsistente: Sesión finalizada.")
            break
        normalized = normalize(query)
        if normalized in {"salir", "exit", "quit"}:
            print("Asistente: Hasta luego.")
            break
        if normalized in {"ayuda", "help"}:
            print(
                "Puedes preguntar:\n"
                "- ¿Qué producto tiene menor presencia en los anaqueles?\n"
                "- ¿Qué productos predominan en la exhibición?\n"
                "- ¿Qué productos faltan en más anaqueles?\n"
                "- ¿Qué anaquel está más lleno?\n"
                "- ¿Cuánto espacio ocupa la leche?\n"
                "- ¿Qué dicen los clientes sobre el cereal?\n"
                "- ¿Qué quejas hay sobre la gaseosa?\n"
                "- ¿Qué producto recibe más comentarios negativos?\n"
                "- ¿Desde hace cuánto falta el detergente?\n"
            )
            continue
        if not query:
            continue

        probabilities = classifier.predict_proba([query])[0]
        class_names = classifier.classes_
        ranked_indices = probabilities.argsort()[::-1]
        best_index = int(ranked_indices[0])
        second_index = int(ranked_indices[1])
        intent = str(class_names[best_index])
        score = float(probabilities[best_index])
        second_score = float(probabilities[second_index])

        if score < MIN_INTENT_SCORE or (score - second_score) < MIN_INTENT_MARGIN:
            first_intent = str(class_names[best_index])
            second_intent = str(class_names[second_index])
            print(
                "Asistente: No puedo identificar la consulta con suficiente seguridad. "
                f"¿Te refieres a '{first_intent}' o a '{second_intent}'? "
                "Reformula la pregunta con más detalle.\\n"
            )
            continue

        response = answer(query, intent, shelves, opinion_by_id)
        print(f"Asistente: {response}")
        print(f"  [Intención detectada: {intent} | puntuación del clasificador: {score:.2f}]\n")


if __name__ == "__main__":
    raise SystemExit(main())
