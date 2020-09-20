from typing import List

import torch
from transformers import DistilBertTokenizerFast 

UNIQUE_TAGS = ['B-Ingredient', 'I-Ingredient', 'B-Product', 'I-Product', 'O']
tag2id = {tag: id for id, tag in enumerate(UNIQUE_TAGS)}
id2tag = {id: tag for tag, id in tag2id.items()}

def flatten(l: List[list]) -> list: 
    return [item for sublist in l for item in sublist]

def pad_list(labels: list, encodings_len) -> list:
    """Takes an individual tag list and pads it to match the length of 
    the token encodings"""
    label_len = len(labels)
    if label_len < encodings_len:
        n_paddings = encodings_len - label_len
        padded = labels + ([0] * n_paddings)
        return padded 
    return labels

def encode_tags(tags: List[str], encodings_len):
    labels = [pad_list([tag2id[tag] for tag in doc], encodings_len) for doc in tags]
    return labels

def get_words_and_labels(docs: list):
    words, labels = [], []
    for doc in docs:
        if not doc:
            continue
        doc = doc.strip()
        lines = doc.split('\n')
        words.append([line.split()[0] for line in lines])
        labels.append([line.split()[-1] for line in lines])
    return words, labels

def detokenize(tokenizer: DistilBertTokenizerFast, words_list: List[list]):
    detokenized = [tokenizer.decode(tokenizer.convert_tokens_to_ids(words)) 
                    for words in words_list]
    return detokenized

def preprocess_ls_data(ls_conll_data, prop_train: float = 0.8):

    """Takes LabelStudio CONLL-formatted data, preprocesses it into training
    data for `transformers`."""

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased', do_lower_case=False)
    docs = ls_conll_data.split('\n\n')[1:]
    n_train_docs = int(len(docs) * prop_train)
    train_docs, val_docs = docs[:n_train_docs], docs[n_train_docs:]

    train_words, train_labels = get_words_and_labels(train_docs)
    val_words, val_labels = get_words_and_labels(val_docs)
    train_seqs, val_seqs = detokenize(tokenizer, train_words), detokenize(tokenizer,val_words)

    train_word_encodings = tokenizer(train_seqs, is_pretokenized=False, return_offsets_mapping=True, padding=True,
     truncation=True, add_special_tokens=False)
    train_label_encodings = encode_tags(train_labels, len(train_word_encodings['input_ids'][0]))

    val_word_encodings = tokenizer(val_seqs, is_pretokenized=False, return_offsets_mapping=True, padding=True,
     truncation=True, add_special_tokens=False)
    val_label_encodings = encode_tags(val_labels, len(val_word_encodings['input_ids'][0]))

    return train_word_encodings, train_label_encodings, val_word_encodings, val_label_encodings

class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.unique_tags = UNIQUE_TAGS

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


