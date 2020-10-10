import json
from typing import List

import torch
from transformers import DistilBertTokenizerFast

UNIQUE_TAGS = ["B-Ingredient", "I-Ingredient", "B-Product", "I-Product", "O"]
tag2id = {tag: id for id, tag in enumerate(UNIQUE_TAGS)}
tag2id_no_prod = dict(zip(UNIQUE_TAGS, [0, 1, 2, 2, 2]))
id2tag = {id: tag for tag, id in tag2id.items()}
id2tag_no_prod = {id: tag for tag, id in tag2id_no_prod.items()}


def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-cased", do_lower_case=False
    )
    return tokenizer


def flatten(l: List[list]) -> list:
    return [item for sublist in l for item in sublist]


def pad_list(labels: list, encodings_len: int) -> list:
    """Takes an individual tag list and pads it to match the length of 
    the token encodings."""
    label_len = len(labels)
    if label_len < encodings_len:
        n_paddings = encodings_len - label_len
        padded = labels + ([0] * n_paddings)
        return padded
    return labels


def encode_tags(tags: List[str], encodings_len: int, tag2id: dict):
    labels = [pad_list([tag2id[tag] for tag in doc], encodings_len) for doc in tags]
    return labels


def get_words_and_labels(docs: list):
    words, labels = [], []
    for doc in docs:
        if not doc:
            continue
        doc = doc.strip()
        lines = doc.split("\n")
        words.append([line.split()[0] for line in lines])
        labels.append([line.split()[-1] for line in lines])
    return words, labels


def detokenize(tokenizer: DistilBertTokenizerFast, words_list: List[list]):
    detokenized = [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(words)) for words in words_list
    ]
    return detokenized


def preprocess_bio_data(
    bio_formatted_data: str, prop_train: float = 0.8, no_product_labels: bool = False
):

    """
    Takes pretokenized CONLL-formatted data, preprocesses it into training
    data for `transformers` model DistilBERT.
    """

    tag2id_dict = tag2id if no_product_labels == False else tag2id_no_prod
    tokenizer = get_tokenizer()
    docs = bio_formatted_data.split("\n\n")
    n_train_docs = int(len(docs) * prop_train)
    train_docs, val_docs = docs[:n_train_docs], docs[n_train_docs:]

    train_words, train_labels = get_words_and_labels(train_docs)
    val_words, val_labels = get_words_and_labels(val_docs)
    train_seqs, val_seqs = (
        detokenize(tokenizer, train_words),
        detokenize(tokenizer, val_words),
    )

    train_word_encodings = tokenizer(
        train_seqs,
        is_pretokenized=False,
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    train_label_encodings = encode_tags(
        train_labels, len(train_word_encodings["input_ids"][0]), tag2id_dict
    )

    val_word_encodings = tokenizer(
        val_seqs,
        is_pretokenized=False,
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    val_label_encodings = encode_tags(
        val_labels, len(val_word_encodings["input_ids"][0]), tag2id_dict
    )

    return (
        train_word_encodings,
        train_label_encodings,
        val_word_encodings,
        val_label_encodings,
    )


def ls_spans_to_bio(ls_data_path: str, save_path: str):

    """
    Standalone function to convert LabelStudio span-formatted data to BIO labels.
    """

    with open(ls_data_path) as f:
        examples = json.load(f)

    tokenizer = get_tokenizer()
    seqs = [example["data"]["text"] for example in examples]
    labels = [example["completions"][0]["result"] for example in examples]
    encodings = tokenizer(
        seqs,
        is_pretokenized=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=True,
        add_special_tokens=False,
    )
    bio_labels = _spans_to_bio(labels, encodings)
    tokens = [enc.tokens for enc in encodings.encodings]

    with open(save_path, "w") as f:
        for toks, labs in zip(tokens, bio_labels):
            lines = [f"{t}\t{l}" for t, l in zip(toks, labs)]
            entry = "\n".join(lines)
            f.write(entry + "\n\n")

    return tokens, bio_labels


def _spans_to_bio(labels: List[List[dict]], encodings):

    """Inner function used by `ls_spans_to_bio`."""

    bio_labels = []

    for enc, label_set in zip(encodings.encodings, labels):
        tok_starts = [tup[0] for tup in enc.offsets]
        tok_ends = [tup[1] for tup in enc.offsets]
        token_labels = ["O"] * len(enc)

        for label in label_set:
            entry = label["value"]
            start, end, ent_type = entry["start"], entry["end"], entry["labels"][0]
            start_token = tok_starts.index(start)
            end_token = tok_ends.index(end) + 1

            # convert to bio format
            if start_token == end_token + 1:
                # Single token is tagged; just throw a B on it
                token_labels[start_token] = f"B-{ent_type}"
            else:
                n_tokens = end_token - start_token
                token_labels[start_token:end_token] = [f"I-{ent_type}"] * n_tokens
                token_labels[start_token] = f"B-{ent_type}"

        bio_labels.append(token_labels)

    return bio_labels


class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.unique_tags = UNIQUE_TAGS

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
