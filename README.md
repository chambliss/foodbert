Next steps:
* Get a big dataset (probably Nom Nom Paleo to start), extract all the ingredients, do EDA
* Document the datasets, labeling process/rules followed. Provide in both DistilBERT BIO tags and LabelStudio span format. 
* Clean up the codebase and push to GitHub
  * Clean up training script
  * Clean up evaluation script
  * Clean up data utils
  * Reorganize functions in functional_model.py
  * Push dataset to GitHub
* Upload model to HuggingFace

### Random notes, for a future blog post
* Data labeling for this task is like playing whack a mole. Had to get data specifically for specific cuisines to shore up performance differences. Model overfit to some entities, extracting only the singular form even when it was plural (mushroom, cereal). 
* Some ingredients / base foods are talked about as if they're products (best example is grape varieties like Cabernet Sauvignon...do you tag that as an ingredient, even though it's capitalized and clearly looks like a product?)
* When should you include modifiers on ingredients? "Roasted almonds" vs. roasted "almonds"? What about "cherry ice cream" vs. "cherry" "ice cream"? How about something like "green onions," where a green onion is clearly different from an onion? Should you tag "garlic cloves" or "garlic" cloves? I admit I probably wasn't completely consistent with this, because every time I come across one, it feels like it needs to be decided on a case by case basis.
* Product is quite a hard class - it's hard to figure out how to tag the span (include the brand name or no?), whether a given span refers to the company or the product, etc
* Including named recipes into products was also questionable. It may be more ideal from a performance perspective to simply break named recipes down into their mentioned ingredients.
* It was useful being able to edit my BIO data on the fly. This allowed me to run some examples through my model, correct them quickly, and throw them into the new training set. 
* If you haven't yet, I would highly recommend learning how to create snippets with your editor and how to use multiple cursor functionality.
* SWITCH TO THE NEW HUGGINGFACE TOKENIZERS WITH OFFSETS, THEY ARE AMAZING AND SAVED ME SO MUCH TIME ON THE MOST TEDIOUS PART OF TOKEN CLASSIFICATION. (span realignment)
* Write functions that will convert between the data formats you will be using frequently (in this case, I wrote functions to convert between BIO and LabelStudio spans since LS was my labeling tool of choice)
* The biggest errors I'm seeing are [[insert results of error analysis here.]]