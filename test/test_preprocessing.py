from food_extractor.data_utils import preprocess_bio_data

example_bio_text = """Past	B-Ingredient
##a	I-Ingredient
carbon	I-Ingredient
##ara	I-Ingredient
.	O
Co	B-Ingredient
##rn	I-Ingredient
##me	I-Ingredient
##al	I-Ingredient
pound	B-Ingredient
##cake	I-Ingredient
.	O

Root	B-Ingredient
vegetables	I-Ingredient
with	O
chick	B-Ingredient
##pe	I-Ingredient
##as	I-Ingredient
and	O
yo	B-Ingredient
##gu	I-Ingredient
##rt	I-Ingredient
.	O
Ch	O
##ees	O
##y	O
s	B-Ingredient
##hak	I-Ingredient
##shu	I-Ingredient
##ka	I-Ingredient
.	O

Sa	B-Ingredient
##rdi	I-Ingredient
##ne	I-Ingredient
-	O
c	B-Ingredient
##ele	I-Ingredient
##ry	I-Ingredient
salad	B-Ingredient
.	O"""


def test_preprocessing():
    train_encodings, train_labels, val_encodings, val_labels = preprocess_bio_data(
        example_bio_text
    )

    tokens = train_encodings[0].tokens
    mask = train_encodings[0].attention_mask
    assert tokens == [
        "Past",
        "##a",
        "carbon",
        "##ara",
        ".",
        "Co",
        "##rn",
        "##me",
        "##al",
        "pound",
        "##cake",
        ".",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
    ]
    assert mask == [1 if tok != "[PAD]" else 0 for tok in tokens]
    assert train_labels[0] == [0, 1, 1, 1, 4, 0, 1, 1, 1, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0]
    assert val_labels == [[0, 1, 1, 4, 0, 1, 1, 0, 4]]
