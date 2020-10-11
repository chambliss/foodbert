import pytest

from food_extractor.food_model import FoodModel, do_preds, UnknownFormatError

test_text = """Banchan-style home cooking is cumulative, which is to say, you might make one or two dishes at a time and keep leftovers in the fridge. 

The point is that you’re amassing a store of banchan so that, come dinnertime, all that’s left to do is steam the rice and take out your stash.

Some banchan can be eaten as soon as you make them. But others are meant to be eaten later, stemming from historic methods of preservation. 

On the Korean Peninsula, food often had to be preserved, especially with salt, to last through the long, grueling winters. 

That’s why fermentation is central to many banchan, like kimchi, pickles and jeotgal, or salted seafood.""".split(
    "\n\n"
)


model = FoodModel("chambliss/distilbert-for-food-extraction")


def test_load_model_local():
    model = FoodModel("../models/model_seed_9")


def test_predict():

    preds = model.predict(test_text)
    ents_only = model.predict(test_text, entities_only=True)

    # Confidence should be high for these examples
    assert all([pred["avg_probability"] > 0.98 for pred in preds])

    # no entities in first example
    assert all([label == "O" for label in preds[0]["labels"]])
    assert not (ents_only[0]["Ingredient"] or ents_only[0]["Product"])

    # returned dicts should match for both
    assert preds[1]["entities"] == ents_only[1]

    # no Products in these examples
    assert all([not ents["Product"] for ents in ents_only])

    # test that all the spans match up
    for ent in ents_only[4]["Ingredient"]:
        (start, end), text = ent["span"], ent["text"]
        assert test_text[4][start:end] == text

test_predict()

def test_predict_single():
    preds = model.predict(test_text[-1])
    assert len(preds) == 1
    ingredients = preds[0]["entities"]["Ingredient"]
    assert all([ent["conf"] > 0.98 for ent in ingredients])
    assert len(ingredients) == 5
    assert not preds[0]["entities"]["Product"]


def test_predict_to_iob():
    preds = model.predict_to_iob(test_text[:2])
    assert (
        preds
        == "Ban\tO\n##chan\tO\n-\tO\nstyle\tO\nhome\tO\ncooking\tO\nis\tO\ncumulative\tO\n,\tO\nwhich\tO\nis\tO\nto\tO\nsay\tO\n,\tO\nyou\tO\nmight\tO\nmake\tO\none\tO\nor\tO\ntwo\tO\ndishes\tO\nat\tO\na\tO\ntime\tO\nand\tO\nkeep\tO\nleft\tO\n##overs\tO\nin\tO\nthe\tO\nfridge\tO\n.\tO\n\nThe\tO\npoint\tO\nis\tO\nthat\tO\nyou\tO\n’\tO\nre\tO\nam\tO\n##ass\tO\n##ing\tO\na\tO\nstore\tO\nof\tO\nban\tB-Ingredient\n##chan\tI-Ingredient\nso\tO\nthat\tO\n,\tO\ncome\tO\ndinner\tO\n##time\tO\n,\tO\nall\tO\nthat\tO\n’\tO\ns\tO\nleft\tO\nto\tO\ndo\tO\nis\tO\nsteam\tO\nthe\tO\nrice\tB-Ingredient\nand\tO\ntake\tO\nout\tO\nyour\tO\ns\tO\n##tas\tO\n##h\tO\n.\tO\n\n"
    )


def test_predict_to_labelstudio():
    preds = model.predict_for_labelstudio(test_text)
    example = preds[1]
    predictions = example["predictions"][0]
    assert [item in example.keys() for item in ["data", "predictions"]]
    assert [item in predictions.keys() for item in ["result", "score"]]
    assert example["data"]["text"] == test_text[1]
    assert [
        item in predictions["result"][0].keys()
        for item in ["from_name", "to_name", "type", "value"]
    ]
    entities = [item["value"] for item in predictions["result"]]
    assert entities == [
        {"end": 52, "labels": ["Ingredient"], "start": 45, "text": "banchan"},
        {"end": 118, "labels": ["Ingredient"], "start": 114, "text": "rice"},
    ]


def test_extract_foods():
    bad_text = "These Are Some Gnarly Carrots Dude!"
    preds = model.extract_foods(bad_text)[0]
    assert not (preds["Product"] or preds["Ingredient"])


def test_do_preds():

    # test that it runs without issue (not going to test the full output)
    do_preds("chambliss/distilbert-for-food-extraction", test_text, "./whatever.json", format="json")

    # test that it raises an error if an incorrect format is specified
    with pytest.raises(UnknownFormatError):
        do_preds(
            "chambliss/distilbert-for-food-extraction", test_text, "./whatever.json", format="butterflies"
        )
