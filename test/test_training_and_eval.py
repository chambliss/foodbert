import os
from food_extractor.train import train


def test_training():
    train_data_path = "../data/training/train_04.txt"
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

    assert os.path.exists("../models/model_seed_9")

    eval_dir = "../data/performance/model_seed_9/"
    artifacts = [
        "eval_percentages.csv",
        "eval_PRF1.csv",
        "eval_raw_stats.csv",
        "preds.log",
    ]
    files_in_eval_dir = os.listdir(eval_dir)
    assert all([artifact in files_in_eval_dir for artifact in artifacts])


test_training()
