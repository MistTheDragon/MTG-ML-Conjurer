import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Load the slimmed-down scryfall oracle text file and convert to pandas DataFrame
#df = pd.read_json("MTG-ML-Conjurer\src\scryfall\oracle-cards-trim.json")

#print(df.head())

# Convert the three elements that are lists to texts and numbers
#df["colors"] = df["colors"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
#df["color_identity"] = df["color_identity"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
#df["keywords"] = df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

#df = df.fillna("")


# First we train a model just on cards with creature typing
df = pd.read_json("MTG-ML-Conjurer\src\scryfall\oracle-cards-trim.json")

# Remove cards that aren't creatures
df = df[df["power"].notna() & df["toughness"].notna()]

df = df.fillna("")

# Build input text
df["input"] = (
    "POWER=" + df["power"].astype(str) + " " +
    "TOUGHNESS=" + df["toughness"].astype(str)
)

# Build output text
df["output"] = (
    "MANA=" + df["mana_cost"]
)

# Keep only what we need
training = df[["input", "output"]]
training.to_csv("PT_train.csv", index=False, encoding="utf-8")
#training.to_csv("creature_train.csv", index=False, encoding="utf-8")


# Save Graveyard
#"KEYWORDS=" + df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") (line 36)