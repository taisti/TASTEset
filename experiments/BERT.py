import argparse
import re
import json
import numpy as np
from transformers import (BertForTokenClassification, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForTokenClassification,
                          set_seed)
from datasets import Dataset
from sklearn.model_selection import KFold
from utils import evaluate_predictions, prepare_data, ENTITIES


LABEL2ID = {"O": 0}
idx = 1
for entity in ENTITIES:
    LABEL2ID[f"B-{entity}"] = idx
    idx += 1
    LABEL2ID[f"I-{entity}"] = idx
    idx += 1

CONFIG = {
    "bert_type": None,
    "model_name_or_path": None,
    "num_of_tokens": 128,
    "only_first_token": True,

    "training_args": {
        "output_dir": './bert-checkpoints',
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 10,
        "weight_decay": 0.01,
    },

    "label2id" : LABEL2ID
}


def check_if_entity_correctly_began(entity, prev_entity):
    """
    This function checks if "I-" entity is preceded with "B-" or "I-". For
    example, "I-FOOD" should not happen after "O" or after "B-QUANT".
    :param entity:
    :param prev_entity:
    :return: bool
    """
    if "I-" in entity and re.sub(r"[BI]-", "", entity) != \
            re.sub(r"[BI]-", "", prev_entity):
        return False
    return True


def token_to_entity_predictions(text_split_words, text_split_tokens,
                                token_labels, id2label):
    """
    Transform token (subword) predictions into word predictions.
    :param text_split_words: list of words from one recipe, eg. ["I", "eat",
    "chicken"] (the ones that go to tokenizer)
    :param text_split_tokens: list of tokens from one recipe, eg. ["I", "eat",
    "chic", "##ken"] (the ones that arise
    from input decoding)
    :param token_labels: list of labels associated with each token from
    text_split_tokens
    :param id2label: a mapping from ids (0, 1, ...) to labels ("B-FOOD",
    "I-FOOD", ...)
    :return: a list of entities associated with each word from text_split_words,
    ie. entities extracted from a recipe
    """

    word_idx = 0
    word_entities = []
    word_from_tokens = ""
    word_entity = ""
    prev_word_entity = ""

    for token_label, token in zip(token_labels, text_split_tokens):
        if token in ["[SEP]", "[CLS]"]:
            continue
        word_from_tokens += re.sub(r"^##", "", token)
        # take the entity associated with the first token (subword)
        word_entity = id2label[token_label] if word_entity == "" \
            else word_entity

        if word_from_tokens == text_split_words[word_idx] or\
                word_from_tokens == "[UNK]":
            word_idx += 1
            # replace entities containing "I-" that do not have a predecessor
            # with "B-"
            word_entity = "O" if not \
                check_if_entity_correctly_began(word_entity, prev_word_entity) \
                else word_entity
            word_entities.append(word_entity)
            word_from_tokens = ""
            prev_word_entity = word_entity
            word_entity = ""

    return word_entities


def tokenize_and_align_labels(recipes, entities, tokenizer, max_length,
                              label2id, only_first_token=True):
    """
    :param recipes: list of lists of words from a recipe
    :param entities: list of lists of entities from a recipe
    :param tokenizer: tokenizer
    :param max_length: maximal tokenization length
    :param label2id: a mapping from labels ("B-FOOD", "I-FOOD", ...) to ids
    (0, 1, ...)
    :param only_first_token: whether to label only first subword of a word,
    eg. Suppose "chicken" is split into "chic", "##ken". Then if True, it will
    have [1, -100], if False [1, 1]. -100
    is omitted in Pytorch loss function
    :return: a dictionary with tokenized recipes with/without associated token
    labels
    """
    tokenized_data = tokenizer(recipes, truncation=True, max_length=max_length,
                               is_split_into_words=True)

    if entities:
        labels = []
        for i, entity in enumerate(entities):
            # Map tokens to their respective word.
            word_ids = tokenized_data.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    new_label = -100
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    new_label = label2id[entity[word_idx]]
                else:
                    new_label = -100 if only_first_token \
                        else label2id[entity[word_idx]]
                label_ids.append(new_label)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_data["labels"] = labels
    tokenized_data["recipes"] = recipes

    return tokenized_data


class TastyModel:
    def __init__(self, config):

        self.config = config
        bert_type = self.config['bert_type']
        model_name_or_path = self.config["model_name_or_path"] if \
            self.config["model_name_or_path"] is not None else bert_type

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

        label2id = {k: int(v) for k, v in self.config["label2id"].items()}
        id2label = {v: k for k, v in label2id.items()}

        # for reproducibility
        set_seed(self.config["training_args"]["seed"])

        model = BertForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(self.config["label2id"]),
            ignore_mismatched_sizes=True,
            label2id=label2id, id2label=id2label
        )

        training_args = TrainingArguments(
            **self.config["training_args"]
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            max_length=self.config["num_of_tokens"],
            padding="max_length"
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

    def train(self, train_recipes, train_entities):

        _, train_dataset = self.prepare_data(train_recipes, train_entities)

        self.trainer.train_dataset = train_dataset

        self.trainer.train()


    def evaluate(self, recipes, entities):

        pred_entities = self.predict(recipes)

        results = evaluate_predictions(entities, pred_entities, "bio")

        return results

    def predict(self, recipes):

        data, dataset = self.prepare_data(recipes, [])
        preds = self.trainer.predict(dataset)

        token_probs = preds[0]
        token_labels = token_probs.argmax(axis=2)

        pred_entities = []

        num_of_recipes = dataset.num_rows

        for recipe_idx in range(num_of_recipes):
            text_split_words = recipes[recipe_idx]
            text_split_tokens = self.tokenizer.convert_ids_to_tokens(
                data["input_ids"][recipe_idx])

            pred_entities.append(token_to_entity_predictions(
                text_split_words,
                text_split_tokens,
                token_labels[recipe_idx],
                self.trainer.model.config.id2label
            ))

        return pred_entities

    def prepare_data(self, recipes, entities):
        data = tokenize_and_align_labels(
            recipes=recipes, entities=entities, tokenizer=self.tokenizer,
            label2id=self.trainer.model.config.label2id,
            max_length=self.config["num_of_tokens"],
            only_first_token=self.config["only_first_token"]
        )

        dataset = Dataset.from_dict(data)

        return data, dataset


def cross_validate(args):

    bio_recipes, bio_entities = prepare_data("../data/TASTEset.csv", "bio")

    CONFIG["bert_type"] = args.bert_type
    CONFIG["model_name_or_path"] = args.model_name_or_path
    CONFIG["training_args"]["seed"] = args.seed

    kf = KFold(n_splits=args.num_of_folds, shuffle=True, random_state=args.seed)
    cross_val_results = {}

    for fold_id, (train_index, test_index) in enumerate(kf.split(bio_entities)):
        tr_recipes, vl_recipes = [bio_recipes[idx] for idx in train_index], \
                                 [bio_recipes[idx] for idx in test_index]
        tr_entities, vl_entities = [bio_entities[idx] for idx in train_index], \
                                   [bio_entities[idx] for idx in test_index]

        model = TastyModel(config=CONFIG)
        model.train(tr_recipes, tr_entities)
        results = model.evaluate(vl_recipes, vl_entities)

        cross_val_results[fold_id] = results

    with open("bert_cross_val_results.json", "w") as json_file:
        json.dump(cross_val_results, json_file, indent=4)

    # aggregate andprint results
    cross_val_results_aggregated = {
        entity: {"precision": [], "recall": [], "f1": []} for entity in
        ENTITIES + ["all"]
    }

    print(f"{'entity':^20s}{'precision':^14s}{'recall':^14s}{'f1-score':^14s}")
    for entity in cross_val_results_aggregated.keys():
        print(f"{entity:^20s}", end="")
        for metric in cross_val_results_aggregated[entity].keys():
            for fold_id in range(args.num_of_folds):
                cross_val_results_aggregated[entity][metric].append(
                    cross_val_results[fold_id][entity][metric]
                )

            mean = np.mean(cross_val_results_aggregated[entity][metric])
            mean = int(mean * 1000) / 1000
            std = np.mean(cross_val_results_aggregated[entity][metric])
            std = int(std * 1000) / 1000 + 0.001 * \
                  round(std - int(std * 1000) / 1000)
            print(f"{mean:^2.3f} +- {std:^2.3f}", end="")
        print()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bert-type', type=str, required=True,
                        help='BERT type')
    parser.add_argument('--model-name-or-path', type=str,
                        help='path to model checkpoint')
    parser.add_argument('--tasteset-path', type=str,
                        default="../data/TASTEset.csv", help="path to TASTEset")
    parser.add_argument('--num-of-folds', type=int, default=5,
                        help="Number of folds in cross-validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    args = parser.parse_args()

    cross_validate(args)
