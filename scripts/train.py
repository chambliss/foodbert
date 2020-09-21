import torch
from transformers import DistilBertForTokenClassification, TrainingArguments, Trainer

from data_utils import preprocess_ls_data, TokenClassificationDataset
from eval_utils import evaluate_model

TRAIN_DATA_PATH = "../data/train_03.txt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NO_PRODUCT_TAGS = False

# Script
with open(TRAIN_DATA_PATH) as f:
    data = f.read()

train_encodings, train_labels, val_encodings, val_labels = preprocess_ls_data(data, 
                                                    no_product_tags=NO_PRODUCT_TAGS)
train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = TokenClassificationDataset(train_encodings, train_labels)
val_dataset = TokenClassificationDataset(val_encodings, val_labels)

if NO_PRODUCT_TAGS:
    train_dataset.unique_tags = ['B-Ingredient', 'I-Ingredient', 'O']
    val_dataset.unique_tags = ['B-Ingredient', 'I-Ingredient', 'O']

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(train_dataset.unique_tags))
model.to(DEVICE)

MODEL_SAVE_PATH = "../models/model_06_seed_11"

training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    do_eval=True,
    evaluate_during_training=True,
    eval_steps=10,
    warmup_steps=50,                
    weight_decay=0.01,               # strength of weight decay
    overwrite_output_dir=True,
    seed=11
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model(MODEL_SAVE_PATH)

# Runs evaluation and saves a bunch of stats
evaluate_model(MODEL_SAVE_PATH, no_product_labels=NO_PRODUCT_TAGS)