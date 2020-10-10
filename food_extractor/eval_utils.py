from collections import Counter
import json
import logging
import os
from typing import List

import pandas as pd

from food_extractor.food_model import FoodModel

logger = logging.getLogger("errors")


def get_other_true_labels(excluded_tag: str, true_label_dict: dict) -> list:
    """
    Gets the true labels from classes other than the excluded class.
    For calculating number of "total overlap but misclassified" labels.
    """
    other_true_labels = []
    for k in true_label_dict:
        if k != excluded_tag:
            other_true_labels.extend(true_label_dict[k])
    return other_true_labels


def is_between(x, y, z):
    """Calculates whether x is between y and z."""
    return y <= x <= z


# It is possible for some tags to satisfy more than one of these, therefore
# it's important to return after getting one True result rather than counting all Trues.
def is_completely_inside(ML, TL):
    """Returns a bool for whether the model label is inside the true label.
    (Starts too late and ends too late.)"""
    cond_1 = is_between(ML["start"], TL["start"], TL["end"])
    cond_2 = is_between(ML["end"], TL["start"], TL["end"])
    return cond_1 and cond_2


def engulfs_true_label(ML, TL):
    """Returns a bool for whether the true label is inside the model label.
    (Starts too early, ends too late.)"""
    cond_1 = is_between(TL["start"], ML["start"], ML["end"])
    cond_2 = is_between(TL["end"], ML["start"], ML["end"])
    return cond_1 and cond_2


def starts_early_ends_early(ML, TL):
    """Returns a bool for whether the model label starts earlier or at 
    the same time as the true label but ends too early."""
    cond_1 = ML["start"] <= TL["start"]
    cond_2 = is_between(ML["end"], TL["start"], TL["end"])
    return cond_1 and cond_2


def starts_late_ends_late(ML, TL):
    """Returns a bool for whether the model label starts later or at 
    the same time as the true label but ends too late."""
    cond_1 = is_between(ML["start"], TL["start"], TL["end"])
    cond_2 = ML["end"] >= TL["end"]
    return cond_1 and cond_2


def is_overlap(ML: dict, true_labels: list) -> bool:

    """
    Calculate whether this model label has a span overlap with a 
    true label. For calculating partially overlapped labels.
    """

    partial_overlap_conds = [
        is_completely_inside,
        engulfs_true_label,
        starts_early_ends_early,
        starts_late_ends_late,
    ]

    for TL in true_labels:
        for cond in partial_overlap_conds:
            if cond(ML, TL):
                logger.info(
                    f"Partial overlap: {cond.__name__}. Predicted "
                    f"[{ML['text']}], actual entity was [{TL['text']}]"
                )
                return True

    return False


def count_misses(tag, model_labels: list, true_labels: list) -> int:

    """Count misses for this tag. Misses are strict and don't include
    partial matches or misclassified tags."""

    misses = 0
    for label in true_labels[tag]:
        if label not in model_labels[tag]:
            misses += 1
            logger.info(f"Missed entity: {label['text']}")

    return misses


def judge_tags(tag: str, model_labels: dict, true_labels: dict, text: str):

    """
    Judges the model's tags against the annotator-labeled tags
    for a single doc.
    """

    label_classes = []
    true_labels_for_tag = true_labels[tag]
    all_other_true_labels = get_other_true_labels(tag, true_labels)
    other_tag = "Product" if tag == "Ingredient" else "Ingredient"

    for label in model_labels[tag]:

        pred_text = label["text"]

        if label in true_labels_for_tag:
            label_class = "correctly classified, total overlap"

        elif label in all_other_true_labels:
            label_class = "misclassified, total overlap"
            logger.info(
                f"Misclassified (total overlap): [{pred_text}] "
                f"Model said {tag}, was actually {other_tag}"
            )

        elif is_overlap(label, true_labels_for_tag):
            label_class = "correctly classified, partial overlap"

        elif is_overlap(label, all_other_true_labels):
            label_class = "misclassified, partial overlap"

        else:
            label_class = "not a named entity"
            logger.info(f"Not a named entity: [{pred_text}] (label: {tag})")

        label_classes.append(label_class)

    # Convert to dict of counts
    results = Counter(label_classes)
    results["missed"] = count_misses(tag, model_labels, true_labels)

    return results


def judge_perf(perf_dict: dict):

    perf = []
    ent_types = perf_dict.keys()

    for ent in ent_types:

        missed = perf_dict[ent]["missed"]
        prec_denom = sum(perf_dict[ent].values()) - missed
        rec_denom = sum(perf_dict[ent].values()) - perf_dict[ent]["not a named entity"]

        if prec_denom == 0:
            prec_strict, prec_loose = 0, 0
        else:
            prec_strict = (
                perf_dict[ent]["correctly classified, total overlap"] / prec_denom
            )
            prec_loose = (
                perf_dict[ent]["correctly classified, total overlap"]
                + perf_dict[ent]["correctly classified, partial overlap"]
            ) / prec_denom

        rec_strict = perf_dict[ent]["correctly classified, total overlap"] / rec_denom
        rec_loose = (
            perf_dict[ent]["correctly classified, total overlap"]
            + perf_dict[ent]["correctly classified, partial overlap"]
        ) / rec_denom

        perf.append([prec_strict, prec_loose, rec_strict, rec_loose])

    perf_df = pd.DataFrame(perf)
    perf_df.index = ent_types
    perf_df.columns = ["p_strict", "p_loose", "r_strict", "r_loose"]

    return perf_df.round(3)


def reformat_true_labels(completions: list) -> dict:

    labels: List[dict] = completions[0]["result"]
    reformatted = {"Ingredient": [], "Product": []}

    for label in labels:
        info = label["value"]
        reformatted[info["labels"][0]].append(
            {"start": info["start"], "end": info["end"], "text": info["text"]}
        )

    return reformatted


def reformat_model_labels(entities: dict) -> dict:

    reformatted = {"Ingredient": [], "Product": []}

    for ent_type in entities:
        ents = entities[ent_type]
        for ent in ents:
            start, end = ent["span"]
            reformatted[ent_type].append(
                {"start": start, "end": end, "text": ent["text"]}
            )

    return reformatted


def evaluate_model(
    model_path: str, eval_file_path: str, no_product_labels: bool = False
):

    """
    Standalone function that takes a model path, eval data path, and save 
    directory, and fully evaluates the model's performance on the data.
    The metrics are then saved in a directory under `data/performance/{model_path}`.
    Note that an existing directory with the same name will be overwritten.

    Outputs include: 
    - a CSV of the precision/recall/F1 on each label ("eval_PRF1.csv"),
    - raw counts of which mistake types were made ("eval_raw_stats.csv"),
    - the raw counts in percentage format ("eval_percentages.csv"),
    - a log file enumerating each example and the model's mistakes on the example
    ("preds.log")
    """

    model_dir = model_path.split("/")[-1]
    performance_dir = "../data/performance"
    eval_destination = os.path.join(performance_dir, model_dir)
    if not os.path.exists(eval_destination):
        os.makedirs(eval_destination)

    with open(os.path.join("../data", eval_file_path)) as f:
        examples = json.load(f)

    # Set up logging for model errors
    logging.basicConfig(
        filename=os.path.join(eval_destination, "preds.log"),
        format="%(message)s",
        filemode="w",
    )
    logger.setLevel("INFO")

    model = FoodModel(model_path, no_product_labels=no_product_labels)
    tags = ["Ingredient", "Product"]
    perf_dict = {
        "Ingredient": Counter(),
        "Product": Counter(),
    }

    for example in examples:

        text = example["data"]["text"]
        logger.info(f"TEXT: {text}")
        true_labels = reformat_true_labels(example["completions"])
        model_labels = reformat_model_labels(model.extract_foods(text)[0])

        for tag in tags:
            perf_dict[tag] += judge_tags(tag, model_labels, true_labels, text)

        logger.info("")  # Add newline after each example

    # Metrics
    perf_df = pd.DataFrame(judge_perf(perf_dict))
    perf_df.to_csv(os.path.join(eval_destination, "eval_PRF1.csv"))

    # Raw counts of each classification
    raw_stats = pd.DataFrame(perf_dict)
    raw_stats.to_csv(os.path.join(eval_destination, "eval_raw_stats.csv"))

    # Percentages of each classification
    totals_df = raw_stats.drop("missed").T
    totals_df["total_labels"] = totals_df.sum(axis=1)

    for col in totals_df.columns:
        if col != "total_labels":
            totals_df[col] = totals_df[col] / totals_df["total_labels"] * 100.0

    totals_df = totals_df.round(1)
    totals_df.to_csv(os.path.join(eval_destination, "eval_percentages.csv"))
