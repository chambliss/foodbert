import torch
from transformers import DistilBertForTokenClassification, TrainingArguments, Trainer

from food_extractor.data_utils import preprocess_bio_data, TokenClassificationDataset
from food_extractor.eval_utils import evaluate_model


def train(
    train_data_path: str,
    model_save_path: str,
    prop_train: float = 0.8,
    no_product_labels: bool = False,
    seed: int = 9,
    evaluate_after_training: bool = True,
    eval_file_path: str = "../data/eval/eval_labeled.json",
):

    """
    train_data_path: The path to your training data. Will be split 
    model_save_path: The path to where your model should be saved.
    prop_train: The proportion of your training data to be held out for 
    calculating the loss during training.
    no_product_labels: If False, removes Product tags from the training data
    and converts them to O's, so the model will not learn to extract Products.
    seed: Random seed to initialize the weights. I found good results with 9.
    evaluate_after_training: Whether to evaluate the model immediately after
    training and save the stats at `data/performance/{model_path}`.
    eval_file_path: Path to a custom eval file. Note this needs to be a 
    LabelStudio-formatted JSON to work correctly. (See format of included 
    eval file.)
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open(train_data_path) as f:
        data = f.read()

    train_encodings, train_labels, val_encodings, val_labels = preprocess_bio_data(
        data, prop_train=prop_train, no_product_labels=no_product_labels
    )
    train_dataset = TokenClassificationDataset(train_encodings, train_labels)
    val_dataset = TokenClassificationDataset(val_encodings, val_labels)

    if no_product_labels:
        train_dataset.unique_tags = ["B-Ingredient", "I-Ingredient", "O"]
        val_dataset.unique_tags = ["B-Ingredient", "I-Ingredient", "O"]

    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-cased", num_labels=len(train_dataset.unique_tags)
    )
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=7,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        do_eval=True,
        evaluate_during_training=True,
        eval_steps=10,
        warmup_steps=50,
        weight_decay=0.01,  # strength of weight decay
        overwrite_output_dir=True,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(model_save_path)

    # Runs evaluation and saves a bunch of stats
    if evaluate_after_training:
        evaluate_model(
            model_save_path,
            eval_file_path=eval_file_path,
            no_product_labels=no_product_labels,
        )
        print(
            "Model has been evaluated. Results are available at "
            f"../data/performance/{model_save_path.split('/')[-1]}."
        )


if __name__ == "__main__":

    train_data_path = "../data/train_04.txt"
    model_save_path = "../models/model_seed_9"
    prop_train = 0.8
    no_product_labels = False
    seed = 9
    evaluate_after_training = True

    train(
        train_data_path=train_data_path,
        model_save_path=model_save_path,
        prop_train=prop_train,
        no_product_labels=no_product_labels,
        seed=seed,
        evaluate_after_training=evaluate_after_training,
    )
