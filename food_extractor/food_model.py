import json
from typing import List, Union

import numpy as np
import torch
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
)

from food_extractor.data_utils import id2tag, id2tag_no_prod, flatten

HF_MODEL_PATH = "chambliss/distilbert-for-food-extraction"


class FoodModel:
    def __init__(self, model_path: str = HF_MODEL_PATH, no_product_labels: bool = False):

        if model_path == HF_MODEL_PATH:
            self.model = DistilBertForTokenClassification.from_pretrained(
                HF_MODEL_PATH
            )
        else:
            self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer()
        self.label_dict = id2tag if no_product_labels == False else id2tag_no_prod

        self.model.to(self.device)
        self.model.eval()

    def ids_to_labels(self, label_ids: list) -> list:
        return [self.label_dict[tensor.item()] for tensor in label_ids]

    def predict(self, texts: Union[str, List[str]], entities_only: bool = False):

        """
        Predicts token classes on a set of tokens.
        Returns each token, its label, and the probability given to that label.
        Currently just does batch size 0.

        Anecdotally, performance decreases on longer texts.
        (Shorter examples = better performance)
        For best results, try to reduce your inputs to sentences or
        other short pieces of text, rather than paragraphs.
        """

        # Accommodate single-item list
        if type(texts) == str:
            texts = [texts]

        n_examples = len(texts)
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encodings.to(self.device)

        # Convert the logits to probabilities
        logits = self.model.forward(encodings["input_ids"])[0]
        probs_per_token = torch.nn.functional.softmax(logits, dim=2)
        max_probs_per_token = torch.max(probs_per_token, dim=2)
        probs, preds = max_probs_per_token.values, max_probs_per_token.indices
        labels = [self.ids_to_labels(p) for p in preds]

        # Create readable dicts for the various components of each prediction
        # and convert the BIO tag predictions to actual entity spans
        pred_summaries = [
            self.create_pred_summary(encodings[i], labels[i], probs[i])
            for i in range(n_examples)
        ]
        entities = [
            self.process_pred(pred_summary, text)
            for pred_summary, text in zip(pred_summaries, texts)
        ]

        # Return just the entity span dicts if desired, otherwise return
        # the more detailed summaries
        if entities_only:
            return entities

        for pred_summary, ent_dict in zip(pred_summaries, entities):
            pred_summary["entities"] = ent_dict

        return pred_summaries

    def extract_foods(self, text: Union[str, List[str]]) -> dict:

        """
        Wrapper around `.predict` to add some extra quality control.
        Specifically, this function:
        - filters out any Ingredient/Product entities shorter than 3 chars
        - filters out lowercase Products, since correctly-extracted Products 
        should always start with uppercase 
        """

        if type(text) == str:
            text = [text]

        batch_entities = self.predict(text, entities_only=True)

        # Extra quality control; delete any tags 3 chars or shorter
        for entities in batch_entities:
            for ent_type in entities:
                ents = entities[ent_type]

                for ent in ents:
                    if len(ent["text"]) <= 3:
                        entities[ent_type].remove(ent)

                    # Additionally, remove products that start with lowercase
                    # since they are likely to be partial spans
                    if ent_type == "Product":
                        if ent["text"][0].islower():
                            if ent in ents:  # prevents attempting to remove twice
                                entities[ent_type].remove(ent)

        return batch_entities

    def create_pred_summary(self, encoding, labels, probs):

        mask = encoding.attention_mask
        tokens = mask_list(encoding.tokens, mask)
        labels = mask_list(labels, mask)
        offsets = mask_list(encoding.offsets, mask)
        prob_list = mask_list(probs.tolist(), mask)

        pred_summary = {
            "tokens": tokens,
            "labels": labels,
            "offsets": offsets,
            "probabilities": prob_list,
            "avg_probability": np.mean(prob_list),
            "lowest_probability": np.min(prob_list),
        }

        return pred_summary

    def get_lowest_confidence_score(self, entities: dict):
        ents = flatten(entities.values())
        if ents:
            lowest_score = min([ent["conf"] for ent in ents])
        else:
            lowest_score = 0
        return lowest_score

    def create_labelstudio_pred(self, entity: dict, ent_type: str) -> dict:
        start, end = entity["span"]
        pred_skeleton = {
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "labels": [ent_type],
                "text": entity["text"],
            },
        }
        return pred_skeleton

    def predict_to_iob(self, texts: Union[str, List[str]]) -> str:

        """
        Generate model predictions in BIO format instead of span format.
        """

        # Accommodate single-item list
        if type(texts) == str:
            texts = [texts]

        preds = self.predict(texts)

        final_output = ""
        for pred in preds:
            toks, labs = pred["tokens"], pred["labels"]
            lines = [f"{t}\t{l}" for t, l in zip(toks, labs)]
            output = "\n".join(lines) + "\n\n"
            final_output = final_output + output

        return final_output

    def predict_for_labelstudio(self, texts: Union[str, List[str]]):

        """
        Produce predictions in a format that can be directly imported by
        LabelStudio.
        """

        # Accommodate single-item list
        if type(texts) == str:
            texts = [texts]

        preds = self.predict(texts, entities_only=True)

        ls_examples = []
        for text, pred in zip(texts, preds):
            formatted_preds = []
            for ent_type in pred:
                ents = pred[ent_type]
                for ent in ents:
                    formatted_preds.append(self.create_labelstudio_pred(ent, ent_type))

            min_score = self.get_lowest_confidence_score(pred)
            ls_example = {
                "data": {"text": text},
                "predictions": [{"result": formatted_preds, "score": min_score}],
            }
            ls_examples.append(ls_example)

        return ls_examples

    def process_pred(self, pred_summary: dict, orig_str: str) -> dict:

        labels = pred_summary["labels"]
        offsets = pred_summary["offsets"]
        probs = pred_summary["probabilities"]

        # TODO: Ent types should not be hardcoded.
        entities = {"Product": [], "Ingredient": []}

        entity_start, entity_end = None, None
        entity_start_idx, entity_end_idx = None, None

        for i, label in enumerate(labels):

            if label == "O":
                continue

            (
                prev_prefix,
                prev_label_type,
                next_prefix,
                next_label_type,
            ) = get_prev_and_next_labels(i, labels)

            prefix, label_type = label.split("-")

            if prefix == "B":
                # Singular entity - end here and insert
                if next_label_type != label_type or next_prefix != "I":
                    start, end = offsets[i]
                    entity = orig_str[start:end]
                    entities[label_type].append(
                        {"text": entity, "span": [start, end], "conf": probs[i]}
                    )
                else:  # Next label is an I w/ same label type
                    entity_start = offsets[i][0]
                    entity_start_idx = i
                    continue

            if prefix == "I":
                # Next label does not continue the entity
                if (
                    i == 0
                    or prev_prefix == "O"
                    or (prev_prefix in ["B", "I"] and label_type != prev_label_type)
                ):
                    entity_start = offsets[i][0]
                    entity_start_idx = i

                if next_label_type != label_type or next_prefix != "I":
                    # End the entity and insert.
                    entity_end = offsets[i][1]
                    entity_end_idx = i
                    entity = orig_str[entity_start:entity_end]
                    entities[label_type].append(
                        {
                            "text": entity,
                            "span": [entity_start, entity_end],
                            "conf": np.mean(
                                probs[entity_start_idx : entity_end_idx + 1]
                            ),
                        }
                    )
                else:
                    # we're in the middle of an entity
                    continue

        return entities


class UnknownFormatError(Exception):
    pass


def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-cased", do_lower_case=False
    )
    return tokenizer


def mask_list(orig_list: list, mask: list) -> list:
    """
        Applies a mask to a list.
        """
    masked = [item for item, m in zip(orig_list, mask) if m == 1]
    return masked


def get_prev_and_next_labels(idx: int, labels: List[str]):

    is_first = idx == 0
    is_last = idx == (len(labels) - 1)

    if is_first:
        prev_label, prev_prefix, prev_label_type = "O", None, None
    else:
        prev_label = labels[idx - 1]
        if prev_label != "O":
            prev_prefix, prev_label_type = prev_label.split("-")
        else:
            prev_prefix, prev_label_type = "O", None
    if is_last:
        next_label, next_prefix, next_label_type = "O", None, None
    else:
        next_label = labels[idx + 1]
        if next_label != "O":
            next_prefix, next_label_type = next_label.split("-")
        else:
            next_prefix, next_label_type = "O", None

    return (prev_prefix, prev_label_type, next_prefix, next_label_type)


def do_preds(
    model_path: str, examples: List[str], save_path: str, format: str = "json"
):

    """
    Generates model predictions on the input examples to the requested format 
    (one of ['json', 'bio', 'labelstudio']).
    """

    model = FoodModel(model_path)

    if format == "json":
        preds = [
            {"text": item, "entities": model.predict(item, entities_only=True)}
            for item in examples
        ]
        with open(save_path, "w") as f:
            json.dump(preds, f)

    elif format == "bio":
        preds = "".join([model.predict_to_iob(item) for item in examples])
        with open(save_path, "w") as f:
            f.write(preds)

    elif format == "labelstudio":
        preds = [model.predict_for_labelstudio(item) for item in examples]
        with open(save_path, "w") as f:
            json.dump(preds, f)

    else:
        msg = f"Didn't recognize this format: {format}"
        "Recognized formats are ['json', 'bio', 'labelstudio']."
        raise UnknownFormatError(msg)
