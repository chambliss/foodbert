# FoodBERT: Food Extraction with DistilBERT

## The first-ever deep learning model for automatic food detection and extraction!*

\* (to my knowledge, as of Oct 2020)

## Quickstart

### Setup

1. Clone the repo

    ```bash
    git clone git@github.com:chambliss/foodbert.git
    ```

2. Set up and activate the environment

    ```bash
    cd foodbert
    conda env create -f environment.yml
    conda activate hf-nlp
    ```

3. Pip install the modules
   
    ```bash
    pip install -e .
    ```

### Load the trained model from the `transformers` model zoo

Loading the trained model from HuggingFace can be done in a single line:
```python
from food_extractor.food_model import FoodModel
model = FoodModel("chambliss/distilbert-for-food-extraction")
```

This downloads the model from HF's S3 bucket and means you will always be using the best-performing/most up-to-date version of the model.

You can also load a model from a local directory using the same syntax.



### Extract foods from some text

The model is especially good at extracting ingredients from lists of recipe ingredients, since there are many training examples of this format:

```python
>>> examples = """3 tablespoons (21 grams) blanched almond flour
... ¾ teaspoon pumpkin spice blend
... ⅛ teaspoon baking soda
... ⅛ teaspoon Diamond Crystal kosher salt
... 1½ tablespoons maple syrup or 1 tablespoon honey
... 1 tablespoon (15 grams) canned pumpkin puree
... 1 teaspoon avocado oil or melted coconut oil
... ⅛ teaspoon vanilla extract
... 1 large egg""".split("\n")

>>> model.extract_foods(examples[0])
[{'Product': [], 'Ingredient': [{'text': 'almond flour', 'span': [34, 46], 'conf': 0.9803279439608256}]}]

>>> model.extract_foods(examples)
[{'Product': [], 'Ingredient': [{'text': 'almond flour', 'span': [34, 46], 'conf': 0.9803279439608256}]}, 
{'Product': [], 'Ingredient': [{'text': 'pumpkin spice blend', 'span': [11, 30], 'conf': 0.8877270460128784}]}, 
{'Product': [], 'Ingredient': [{'text': 'baking soda', 'span': [11, 22], 'conf': 0.89898481965065}]}, 
{'Product': [{'text': 'Diamond Crystal kosher salt', 'span': [11, 38], 'conf': 0.7700592577457428}], 'Ingredient': []}, 
... (further results omitted for brevity)
]
```

It also works well on standard prose:
```python
>>> text = """Swiss flavor company Firmenich used artificial intelligence (AI) in partnership with Microsoft to optimize flavor combinations and create a lightly grilled beef taste for plant-based meat alternatives, according to a release."""

>>> model.extract_foods(text)
[{'Product': [], 
'Ingredient': [{'text': 'beef', 'span': [156, 160], 'conf': 0.9615312218666077}, 
{'text': 'plant', 'span': [171, 176], 'conf': 0.8789700269699097}, 
{'text': 'meat', 'span': [183, 187], 'conf': 0.9639666080474854}]}]
```

To get raw predictions, you can also use `model.predict` directly. But note that `extract_foods` has a couple of heuristics added to remove low-quality predictions, so `model.predict` is likely to give slightly worse performance.

That said, it is useful for examining the raw labels/probabilities/etc. from the forward pass.

```python
# Using the same text as the previous example
>>> predictions = model.predict(text)[0]

# All data available from the example
>>> predictions.keys()
dict_keys(['tokens', 'labels', 'offsets', 'probabilities', 'avg_probability', 'lowest_probability', 'entities'])

>>> for t, p in zip(predictions['tokens'], predictions['probabilities']):
...     print(t, round(p, 3))
Swiss 0.991
flavor 0.944
company 0.998
Fi 0.952
...

# Get the token the model was least confident in predicting
>>> least_confident = predictions['probabilities'].index(predictions['lowest_probability'])
>>> predictions[0]['tokens'][least_confident]
'plant'

# Get the dict of ingredients and products
>>> predictions['entities']
{'Product': [],
 'Ingredient': [{'text': 'beef',
   'span': [156, 160],
   'conf': 0.9615312218666077},
   ...

```


### Larger-scale prediction

To predict on many examples, you can use `food_model.do_preds`. I usually use this for generating model predictions to correct in [LabelStudio](https://labelstud.io/), the tagging platform used for this project. Calling it looks like this:
```python
from food_extractor.food_model import do_preds

do_preds("chambliss/distilbert-for-food-extraction", # model path
        texts, # your examples - a list of strings
        "./whatever.json", # output file
        format="json") # format - JSON, LabelStudio, and BIO are supported
```

Note that this will run each example through the model individually rather than batching. This results in better performance (the model is more confident in its predictions on shorter sequences, so by not having to pad the examples to be the same length as the longest example in the batch, the accuracy is increased slightly). 

Since we're using DistilBERT, prediction is still very fast (especially on GPU), but if you prefer to batch the examples, it should be relatively easy to amend the code in `do_preds` to do so.

Also note that these are **raw predictions** from the model, not the quality-filtered predictions you will get from `extract_foods`. 

### Model Stats


**Model Type**: DistilBERT, base-cased

**Model Size**: 260.8MB

**Inference Time**: 0.03s for batch size 1 on CPU, 0.06s for batch size 9 on CPU

**Performance**

The model performs best on the Ingredient tag, reaching over 90% relaxed precision and over 75% relaxed recall. Products were not common in the training data, and thus have significantly worse performance. 

If you have a production use case in mind for this, the model should perform well enough (with some data cleaning) to systematically extract ingredients, but I would not recommend using the Product results for production use cases at the moment.

|Label    |p_strict|p_loose|r_strict|r_loose|
|----------|--------|-------|--------|-------|
|Ingredient|0.787   |0.912  |0.681   |0.789  |
|Product   |0.211   |0.649  |0.171   |0.529  |

Description of metrics:
* p_strict: Strict, exact-match precision. 
* p_loose: Relaxed precision, where "partial overlap" errors are allowed. For this task, it is usually more useful to look at relaxed precision rather than strict.
* r_strict: Strict, exact-match recall. For products in particular, it is probably most useful to look at strict recall.
* r_loose: Relaxed recall, where if part of an ingredient/product was retrieved, it was counted.

Quick example to clarify the difference between **strict** and **loose** precision and recall:

* If the model predicted "blanched almond flour," but the actual ingredient label was "almond flour," this would count AGAINST strict precision, but it would be an allowable prediction for loose precision. 
* Similarly, if the actual product was "Raspberry Red ice cream" and the model only predicted "Red ice cream," this would be allowable for measuring loose recall, but it would NOT count for strict recall. 

----
## Training and Evaluation Data

### Training 

The model was trained on 715 examples, most of them on the shorter side (many were extracted ingredient entries from web-scraped recipes). The data is BIO-formatted (begin-inside-outside), and looks like this:
```text
G       B-Ingredient
##ar    I-Ingredient
##lic   I-Ingredient
is      O
extremely       O
healthy O
and     O
can     O
be      O
used    O
in      O
a       O
variety O
of      O
recipes O
.       O
```

The training data is small enough to be included in this repo, but of course you should not store any future training data with git. Ideally, use a data version control system such as DVC. 

### Evaluation 

The evaluation data is provided in [LabelStudio](https://labelstud.io/) format, because that is what I used to label it. (I would highly recommend LS for solo labeling projects, by the way.) It has 138 examples and looks like this:

```text
[
  {
    "completions": [
      {
        "created_at": 1600712341,
        "honeypot": true,
        "id": 1209001,
        "lead_time": 10.012,
        "result": [
          {
            "from_name": "label",
            "id": "AnlIUSC81r",
            "parent_id": null,
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 12,
              "labels": [
                "Ingredient"
              ],
              "start": 9,
              "text": "oil"
            }
          },
          ...
```

If you want to import it directly into your own [LabelStudio](https://labelstud.io/) project, this is the config I used in my project:
```xml
<View>
  <Labels name="label" toName="text">
    <Label value="Ingredient" background="#5CC8FF"/>
    <Label value="Product" background="#7D70BA"/>
</Labels>

  <Text name="text" value="$text"/>
</View>
```

### Labeling Rules

Labeling for this task was surprisingly difficult, but there are a few rules that I tried to abide by.

* Ingredients should be stripped down to their basic form. For example, prefer "almond flour" over "blanched almond flour." 
* Avoid including modifiers unless it would result in information loss (prefer "crimini mushrooms" over just "mushrooms," for example).
* Labeled products should include both the brand name and the actual food, for example "CLIF energy bars" rather than just "CLIF."
* Labeling parts of words was allowed, for example "[plant]-based" or "[meat]less."

----

## Going Further

### Training your own model

You can easily train a new model or fine-tune this one using the [training script](https://github.com/chambliss/foodbert/blob/master/food_extractor/train.py#L8). You will need to label some data and convert it to BIO format. A utility function for converting LabelStudio data to BIO format is provided in the [data_utils](https://github.com/chambliss/foodbert/blob/master/food_extractor/data_utils.py#L112) module.

### Evaluating a model

I've created a set of evaluation utilities [eval_utils.py](https://github.com/chambliss/foodbert/blob/master/food_extractor/eval_utils.py#L216) that can do a comprehensive evaluation for you. From the `eval_utils.evaluate_model` definition:
```python
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
```

The `preds.log` file looks like this, and makes it easier to qualitatively understand what kinds of mistakes your model is making.
```text
TEXT: 1.Cut leaf from stem of green bok choy. Cut leaves into length of about 5 cm. Cut stem lengthwise into six equal parts and immerse in water separately. If dirt is found in the root part of stem, scrape out with tip of bamboo skewer (PHOTO A). Remove core from garlic and slice thinly.
Partial overlap: engulfs_true_label. Predicted [green bok choy], actual entity was [bok choy]
Not a named entity: [bamboo] (label: Ingredient)
Missed entity: bok choy
Missed entity: water
```

### Developing on the code

If you want to make changes to the code and ensure that things still work, I've included some basic tests to run that will surface any major errors with training, evaluation, data preprocessing, or generating predictions from the model. 

They are not meant to be comprehensive, so there may still be some "silent failures" you'll need to debug yourself, but they should be a good first line of defense against potentially breaking changes.

To run the tests, make sure `pytest` is installed, then run `pytest test` from the root directory.
