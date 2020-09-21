import json
import os
from typing import List

import numpy as np
import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from data_utils import id2tag, id2tag_no_prod, flatten

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "../models/model_07_seed_9/"
data_to_label = "../data/raw/curated_examples_03.txt"
labeled_save_path = "../data/raw/curated_examples_03_iob.txt"

class FoodModel:

    def __init__(self, model_path: str, no_product_labels=False):
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased', do_lower_case=False)
        self.label_dict = id2tag if no_product_labels == False else id2tag_no_prod

    def ids_to_labels(self, label_ids: list) -> list:
        return [self.label_dict[tensor.item()] for tensor in label_ids]

    def predict(self, text: str, entities_only=False):

        """
        Predicts token classes on a set of tokens.
        Returns each token, its label, and the probability given to that label.
        Currently just does batch size 0.
        """

        encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encodings.to(device)

        # model.forward also returns (optional) loss, hidden_states, attentions
        outputs = self.model.forward(encodings['input_ids'])[0]
        logits = outputs[0, :, :] # again, batch size 0

        # Convert the logits to probabilities via softmax
        probs_per_token = torch.nn.functional.softmax(logits, dim=1)
        max_probs_per_token = torch.max(probs_per_token, dim=1)
        # Getting the indices is the same as doing `torch.argmax`
        probs, preds = max_probs_per_token.values, max_probs_per_token.indices
        labels = self.ids_to_labels(preds)
        
        pred_summary = {"tokens": encodings.tokens(), 
                        "labels": labels, 
                        "offsets": encodings[0].offsets,
                        "probabilities": probs.tolist(), 
                        "global_probability": torch.mean(probs).item()}

        entities = self.process_pred(pred_summary, text)
        pred_summary["entities"] = entities

        if entities_only:
            return entities
        return pred_summary

    def extract_foods(self, text: str):
        entities = self.predict(text, entities_only=True)
        
        # Extra quality control; delete any tags 3 chars or shorter
        for ent_type in entities:
            ents = entities[ent_type]
            for ent in ents:
                if len(ent['text']) <= 3:
                    entities[ent_type].remove(ent)

        return entities

    def get_lowest_confidence_score(self, entities: dict):
        ents = flatten(entities.values())
        if ents:
            lowest_score = min([ent['conf'] for ent in ents])
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
                "text": entity["text"]
            }
        }
        return pred_skeleton

    def predict_to_iob(self, text: str) -> str:

        pred = self.predict(text)
        toks, labs = pred["tokens"], pred["labels"]
        lines = [f"{t}\t{l}" for t, l in zip(toks, labs)]
        output = "\n".join(lines) + "\n\n"

        return output
        
    def predict_for_labelstudio(self, text: str):

        """
        Produce predictions in a format that can be directly imported by
        LabelStudio.
        """

        preds = self.predict(text, entities_only=True)
        min_score = self.get_lowest_confidence_score(preds)

        formatted_preds = []
        for ent_type in preds:
            ents = preds[ent_type]
            for ent in ents:
                formatted_preds.append(self.create_labelstudio_pred(ent, ent_type))

        ls_example = {"data": {"text": text},
                     "predictions": [
                         {"result": formatted_preds,
                         "score": min_score}
                    ]
                    }
        
        return ls_example

    def process_pred(self, pred_summary: dict, orig_str: str) -> dict:

        tokens = pred_summary['tokens']
        labels = pred_summary['labels']
        offsets = pred_summary['offsets']
        probs = pred_summary['probabilities']

        # TODO: Ent types should not be hardcoded.
        entities = {"Product": [], "Ingredient": []}

        entity_start, entity_end = None, None
        entity_start_idx, entity_end_idx = None, None
        
        for i, label in enumerate(labels):
            
            if label == "O":
                continue

            prev_label, prev_prefix, prev_label_type, next_label, next_prefix, next_label_type = get_prev_and_next_labels(i, labels)

            prefix, label_type = label.split("-")
            
            if prefix == "B":
                # Singular entity - end here and insert
                if next_label_type != label_type or next_prefix != "I":
                    start, end = offsets[i]
                    entity = orig_str[start:end]
                    entities[label_type].append(
                        {"text": entity, "span": [start, end], "conf": probs[i]}
                    )
                else: # Next label is an I w/ same label type
                    entity_start = offsets[i][0]
                    continue
            
            if prefix == "I":
                # Next label does not continue the entity
                if i == 0 or prev_prefix == "O" or (prev_prefix in ["B", "I"] and label_type != prev_label_type):
                     entity_start = offsets[i][0]

                if next_label_type != label_type or next_prefix != "I": 
                    # End the entity and insert.
                    entity_end = offsets[i][1]
                    entity = orig_str[entity_start: entity_end]
                    entities[label_type].append(
                        {"text": entity, 
                        "span": [entity_start, entity_end], 
                        "conf": np.mean(probs[entity_start_idx: entity_end_idx])}
                    )
                else:
                    # we're in the middle of an entity
                    continue

        return entities

    def evaluate(self):

        """Expects entries in the form of a list of dicts, where each
        dict represents an entity, and has keys "text" and "span."
        (Conf key is optional and not currently used.)

        Example: {'text': 'black tea', 'span': [61, 70], 'conf': 0.9717584827116558}
        """

        print("Not implemented yet!")

        return None


def print_example(pred_result_dict: dict, outfile: str):

    tokens, labels = pred_result_dict['tokens'], pred_result_dict['labels']

    with open(outfile, 'a') as out:
        # print("Global probability:", round(pred_result_dict['global_probability'], 3),
        #         file=out)
        entities = pred_result_dict['entities']
        for ent_type in entities:
            print(ent_type.upper(), file=out)
            for entry in entities[ent_type]:
                print(entry, file=out)
        print(file=out)
        for t, l in zip(tokens, labels):
            print(t, "\t", l, file=out)
        print(file=out)

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

        return (prev_label, prev_prefix, prev_label_type, next_label, next_prefix, next_label_type)

def do_preds(model, examples: List[str], outfile='preds.log'): 
    
    # preds = [{"text": item, "entities": model.predict(item, entities_only=True)} 
    #         for item in examples]

    # preds = [model.predict_for_labelstudio(item) for item in examples]
    # with open(outfile, "w") as f:
    #     json.dump(preds, f)
    
    preds = [model.predict_to_iob(item) for item in examples]
    with open(outfile, "w") as f:
        for pred in preds:
            f.write(pred)
    

    
    # for p in preds: 
    #      print_example(p, outfile=outfile)        


# model = FoodModel(model_path)

# with open("../data/eval_data_examples.txt") as f:
#     sents = f.read().split("\n\n")

# outfile = "../data/eval_labelstudio.json"
# if os.path.exists(outfile):
#     os.remove(outfile)

# do_preds(model, sents, outfile=outfile)
if __name__ == "__main__":
    model = FoodModel(model_path)

    with open(data_to_label) as f:
        sents = f.read().split("\n\n")

    outfile = labeled_save_path
    if os.path.exists(outfile):
        os.remove(outfile)

    do_preds(model, sents, outfile=outfile)