#!/usr/bin/env python3
"""Asistente local por reglas para consultar inventario visual y opiniones."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCTS = {
    "canned": {
        "name": "Conservas",
        "aliases": ("conserva", "conservas", "lata", "latas"),
    },
    "cereal": {
        "name": "Cereal",
        "aliases": ("cereal", "cereales"),
    },
    "cookies": {
        "name": "Galletas",
        "aliases": ("galleta", "galletas"),
    },
    "detergent": {
        "name": "Detergente",
        "aliases": ("detergente", "detergentes", "jabon", "lavado"),
    },
    "juice": {
        "name": "Jugo",
        "aliases": ("jugo", "jugos", "zumo"),
    },
    "milk": {
        "name": "Leche",
        "aliases": ("leche",),
    },
    "soda": {
        "name": "Gaseosa",
        "aliases": ("gaseosa", "gaseosas", "refresco"),
    },
    "water": {
        "name": "Agua",
        "aliases": ("agua", "aguas"),
    },
}


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"El archivo {description} no contiene JSON válido: {path}") from exc


def validate_sources(inventory: dict, opinions: dict) -> tuple[list[dict], dict[str, dict]]:
    shelves = inventory.get("shelves")
    products = opinions.get("productos")
    if not isinstance(shelves, list) or not shelves:
        raise ValueError("El JSON de inventario no contiene una lista 'shelves'.")
    if not isinstance(products, list) or not products:
        raise ValueError("El JSON de opiniones no contiene una lista 'productos'.")

    for shelf in shelves:
        if not isinstance(shelf.get("predicted_counts"), dict):
            raise ValueError("Una entrada de inventario no contiene 'predicted_counts'.")
    opinion_by_id = {
        item.get("product_id"): item for item in products if item.get("product_id")
    }
    return shelves, opinion_by_id


def find_product(query: str) -> str | None:
    normalized = normalize(query)
    matches = []
    for product_id, data in PRODUCTS.items():
        for alias in data["aliases"]:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                matches.append((len(alias), product_id))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def find_capture(query: str) -> str | None:
    match = re.search(r"captura[_ -]?(\d+)", normalize(query))
    if not match:
        return None
    return f"captura_{match.group(1).zfill(3)}"


def find_threshold(query: str) -> int | None:
    normalized = normalize(query)
    match = re.search(r"menos de\s+(\d+)", normalized)
    if match:
        return int(match.group(1))
    return None


def format_inventory(shelf: dict, product_id: str) -> str:
    count = shelf["predicted_counts"].get(product_id, 0)
    product = PRODUCTS[product_id]["name"]
    return f"{shelf['capture_id']}: {product} = {count} posiciones visibles."


def answer_inventory_product(shelves: list[dict], product_id: str,
                             capture_id: str | None) -> str:
    selected = [
        shelf for shelf in shelves
        if capture_id is None or shelf.get("capture_id") == capture_id
    ]
    if not selected:
        available = ", ".join(shelf.get("capture_id", "?") for shelf in shelves)
        return f"No encuentro {capture_id}. Anaqueles disponibles: {available}."
    return "\n".join(format_inventory(shelf, product_id) for shelf in selected)


def answer_inventory_shelf(shelves: list[dict], capture_id: str | None,
                           threshold: int | None) -> str:
    selected = [
        shelf for shelf in shelves
        if capture_id is None or shelf.get("capture_id") == capture_id
    ]
    if not selected:
        available = ", ".join(shelf.get("capture_id", "?") for shelf in shelves)
        return f"No encuentro {capture_id}. Anaqueles disponibles: {available}."

    responses = []
    for shelf in selected:
        counts = shelf["predicted_counts"]
        if threshold is not None:
            below = [
                f"{PRODUCTS[product_id]['name']} ({count})"
                for product_id, count in counts.items()
                if product_id in PRODUCTS and product_id != "empty" and count < threshold
            ]
            detail = ", ".join(below) if below else "ningún producto"
            responses.append(
                f"{shelf['capture_id']}: categorías con menos de {threshold} posiciones: {detail}."
            )
        else:
            detail = ", ".join(
                f"{PRODUCTS[product_id]['name']}={count}"
                for product_id, count in counts.items()
                if product_id in PRODUCTS
            )
            responses.append(f"{shelf['capture_id']}: {detail}.")
    return "\n".join(responses)


def answer_product_opinions(opinion_by_id: dict[str, dict],
                            product_id: str, only_negative: bool) -> str:
    item = opinion_by_id.get(product_id)
    if not item:
        return f"No hay opiniones registradas para {PRODUCTS[product_id]['name']}."

    counts = item.get("sentimientos", {})
    name = item.get("producto", PRODUCTS[product_id]["name"])
    response = (
        f"{name}: {counts.get('positivo', 0)} positivas, "
        f"{counts.get('negativo', 0)} negativas y "
        f"{counts.get('neutral', 0)} neutrales "
        f"({item.get('total_comentarios', 0)} comentarios analizados)."
    )

    if only_negative:
        comments = item.get("comentarios_negativos", [])
        if comments:
            examples = [
                f"- «{entry.get('comentario', '')}»"
                for entry in comments[:3]
            ]
            response += "\nComentarios negativos:\n" + "\n".join(examples)
        else:
            response += "\nNo se clasificaron comentarios como negativos."
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
        f"{item.get('producto', item.get('product_id'))} ({maximum})"
        for item in ranked
        if item.get("sentimientos", {}).get("negativo", 0) == maximum
    ]
    return (
        "Categorías con más comentarios negativos clasificados: "
        + ", ".join(leaders)
        + "."
    )


def respond(query: str, shelves: list[dict],
            opinion_by_id: dict[str, dict]) -> str:
    normalized = normalize(query)
    product_id = find_product(query)
    capture_id = find_capture(query)
    threshold = find_threshold(query)

    opinion_terms = (
        "opinion", "opiniones", "comentario", "comentarios",
        "sentimiento", "sentimientos", "cliente", "clientes",
        "resena", "resenas", "queja", "quejas",
    )
    negative_terms = ("negativo", "negativa", "negativas", "negativos", "queja", "quejas")
    inventory_terms = (
        "inventario", "posicion", "posiciones", "anaquel", "espacio",
        "espacios", "cuanto", "cuanta", "cuantos", "cuantas",
    )

    asks_opinions = any(term in normalized for term in opinion_terms)
    asks_negative = any(term in normalized for term in negative_terms)
    asks_inventory = any(term in normalized for term in inventory_terms)

    if asks_opinions and not product_id:
        return answer_most_negative(opinion_by_id)

    if asks_opinions and product_id:
        return answer_product_opinions(opinion_by_id, product_id, asks_negative)

    if asks_inventory and product_id:
        return answer_inventory_product(shelves, product_id, capture_id)

    if asks_inventory or threshold is not None:
        return answer_inventory_shelf(shelves, capture_id, threshold)

    return (
        "No identifiqué la consulta. Prueba con preguntas como:\n"
        "- ¿Cuántas posiciones de leche hay en captura_031?\n"
        "- ¿Qué productos tienen menos de 3 posiciones en captura_031?\n"
        "- ¿Qué opinan los clientes sobre el cereal?\n"
        "- ¿Qué comentarios negativos hay sobre la gaseosa?\n"
        "- ¿Qué categoría tiene más opiniones negativas?"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Asistente conversacional local para Tienda inteligente."
    )
    parser.add_argument(
        "--inventario", type=Path,
        default=Path("cv-lab/salida_ml/iteraciones/evaluacion_01/inventario_por_foto.json"),
        help="JSON de inventario estimado por el modelo de visión.",
    )
    parser.add_argument(
        "--opiniones", type=Path,
        default=Path("nlp-lab/salida/resumen_sentimientos.json"),
        help="JSON de opiniones clasificado por el laboratorio de PLN.",
    )
    args = parser.parse_args()

    inventory_path = args.inventario if args.inventario.is_absolute() else REPO_ROOT / args.inventario
    opinions_path = args.opiniones if args.opiniones.is_absolute() else REPO_ROOT / args.opiniones

    try:
        inventory = load_json(inventory_path, "inventario")
        opinions = load_json(opinions_path, "resumen de sentimientos")
        shelves, opinion_by_id = validate_sources(inventory, opinions)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print("TIENDA INTELIGENTE – Asistente conversacional")
    print("Consulto inventario visual y opiniones clasificadas.")
    print("Escribe 'ayuda' para ver ejemplos o 'salir' para terminar.\n")

    try:
        while True:
            query = input("Tú: ").strip()
            if normalize(query) in {"salir", "exit", "quit"}:
                print("Asistente: Hasta luego.")
                break
            if normalize(query) in {"ayuda", "help"}:
                print(respond("", shelves, opinion_by_id))
                continue
            if not query:
                continue
            print("Asistente:", respond(query, shelves, opinion_by_id), "\n")
    except (EOFError, KeyboardInterrupt):
        print("\nAsistente: Sesión finalizada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
