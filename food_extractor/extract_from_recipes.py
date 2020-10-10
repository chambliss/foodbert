from collections import Counter
import json
import os
import zipfile

import pandas as pd

from food_extractor.food_model import FoodModel

"""Example script using FoodModel to extract ingredients from JSON-formatted recipes."""

MODEL_PATH = "../models/model_07_seed_9_3"
RECIPES_PATH = "../data/raw/recipes"
RECIPES_ARCHIVE = os.path.join(RECIPES_PATH, "recipes.zip")
EXTRACTED_RECIPES_PATH = os.path.join(RECIPES_PATH, "recipes")
AGGREGATE_DF_SAVE_PATH = os.path.join(RECIPES_PATH, "aggregate_counts.csv")
PER_RECIPE_DF_SAVE_PATH = os.path.join(RECIPES_PATH, "per_recipe_ingredients.csv")
model = FoodModel(MODEL_PATH)

if not os.path.exists(EXTRACTED_RECIPES_PATH):
    # Unzip the recipes
    recipe_zip = zipfile.ZipFile(RECIPES_ARCHIVE)
    recipe_zip.extractall(RECIPES_PATH)

recipe_files = [
    os.path.join(EXTRACTED_RECIPES_PATH, fn)
    for fn in os.listdir(EXTRACTED_RECIPES_PATH)
    if ".json" in fn
]
all_recipes = []
for filename in recipe_files:
    with open(filename) as f:
        all_recipes.append(json.load(f))

# Extract the info
all_ingredients = []
recipe_entries = []

for recipe in all_recipes:
    recipe_title = recipe["title"]
    recipe_ingredients = [
        model.extract_foods(ingredient["sentence"])
        for ingredient in recipe["ingredients"]
    ]

    # Track ingredients per-recipe
    ingredient_texts = []
    ingredient_confs = []

    for ingredients in recipe_ingredients:

        annotations = ingredients["Ingredient"]
        ingredient_texts.extend([ann["text"] for ann in annotations])
        ingredient_confs.extend([ann["conf"] for ann in annotations])

        # Track both ingredients and products in aggregate
        for ent_type in ingredients:
            all_ingredients.extend(ingredients[ent_type])

    recipe_entries.append(
        {
            "title": recipe_title,
            "ingredients": ingredient_texts,
            "confs": ingredient_confs,
        }
    )

# Save
pd.DataFrame(recipe_entries).to_csv(PER_RECIPE_DF_SAVE_PATH, index=False)

counts = Counter([ingredient["text"] for ingredient in all_ingredients])
aggregate_df = pd.DataFrame(counts, index=[0]).T.reset_index()
aggregate_df.columns = ["ingredient", "n_mentions"]
aggregate_df.to_csv(AGGREGATE_DF_SAVE_PATH, index=False)
