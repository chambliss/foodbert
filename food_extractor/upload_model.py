from transformers.modeling_tf_distilbert import TFDistilBertForTokenClassification

from food_model import FoodModel

curr_path = "../models/model_seed_9"
model_path = "../models/distilbert-for-food-extraction"

model = FoodModel(curr_path)
tokenizer = model.tokenizer

model.model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

tf_model = TFDistilBertForTokenClassification.from_pretrained(model_path, from_pt=True)
tf_model.save_pretrained(model_path)
