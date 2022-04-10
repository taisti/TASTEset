# BERT

Simply run:
```commandline
python BERT.py --bert-type bert-base-cased
```
to run 5-fold cross-validation.
All tested BERT versions include:
* bert-base-cased
* bert-large-cased
* [FoodNER](https://github.com/ds4food/FoodNer/blob/master/FoodNER.ipynb) 
  checkpoint (the classification layer is excluded due to a different number 
  of predicted classes classes)
