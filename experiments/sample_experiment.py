from src.utils import prepare_data, evaluate_predictions
from sklearn.model_selection import KFold

SEED = 42
NUM_OF_FOLDS = 5


def main():
    recipes, entities = prepare_data("../data/TASTEset.csv")
    bio_recipes, bio_entities = prepare_data("../data/TASTEset.csv", "bio")

    kf = KFold(n_splits=NUM_OF_FOLDS, shuffle=True, random_state=SEED)

    kf.get_n_splits(recipes)

    for fold_id, (train_index, test_index) in enumerate(kf.split(entities)):
        tr_recipes, vl_recipes = [recipes[idx] for idx in train_index],\
                                 [recipes[idx] for idx in test_index]
        tr_entities, vl_entities = [entities[idx] for idx in train_index],\
                                   [entities[idx] for idx in test_index]

        ### TRAIN  - preferably use the same seed ?
        ### PREDICT
        ### EVALUATE
        #results, results_by_tag = evaluate_predictions(...)

    print("SAMPLE SPANS EVALUATION")
    print(entities[:10])
    results = evaluate_predictions(entities[:10], entities[:10], "spans")
    print(results)

    print("SAMPLE BIO EVALUATION")
    print(bio_entities[:10])
    results = evaluate_predictions(bio_entities[:10], bio_entities[:10], "bio")
    print(results)


if __name__ == "__main__":
    main()
