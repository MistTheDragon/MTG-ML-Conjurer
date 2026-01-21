import ijson
import json
from decimal import Decimal

INPUT_FILE = "MTG-ML-Conjurer\src\scryfall\oracle-cards.json"
OUTPUT_FILE = "MTG-ML-Conjurer\src\scryfall\oracle-cards-trim.json"

FIELDS = [
    "id",
    "name",
    "mana_cost",
    "cmc",
    "type_line",
    "oracle_text",
    "power",
    "toughness",
    "colors",
    "color_identity",
    "keywords",
]

count = 0

with open(INPUT_FILE, "rb") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    parser = ijson.items(infile, "item")

    outfile.write("[\n")
    first = True

    for card in parser:
        # Had to normalize because TypeError: Object of type Decimal is not JSON serializable
        def normalize(value):
            if isinstance(value, Decimal):
                return float(value)
            return value

        slim = {k: normalize(card.get(k)) for k in FIELDS}

        if not first:
            outfile.write(",\n")
        else:
            first = False

        json.dump(slim, outfile, ensure_ascii=False)
        count += 1

    outfile.write("\n]")

print(f"Done! Wrote {count} cards.")
