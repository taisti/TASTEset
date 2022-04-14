import argparse
import re
import json
import torch
import numpy as np
from transformers import (BertForTokenClassification, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForTokenClassification,
                          set_seed, BertConfig)
from transformers.utils import ModelOutput
from typing import Optional
from datasets import Dataset
from sklearn.model_selection import KFold
from utils import evaluate_predictions, prepare_data, ENTITIES
from torchcrf import CRF


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

    "label2id": LABEL2ID
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

    labels = []
    recipes_words_beginnings = []  # mark all first subwords,
    # e.g. 'white sugar' which is split into ["wh", "##ite", "sug". "##ar"]
    # would have [1, 0, 1, 0]. This is used as prediction mask in the BertCRF

    for recipe_idx in range(len(recipes)):
        # Map tokens to their respective word.
        word_ids = tokenized_data.word_ids(batch_index=recipe_idx)
        previous_word_idx = None
        label_ids = []
        words_beginnings = []
        for word_idx in word_ids:
            if word_idx is None:
                words_beginnings.append(False)
            elif word_idx != previous_word_idx:
                words_beginnings.append(True)
            else:
                words_beginnings.append(False)
            if entities:
                if word_idx is None:
                    new_label = -100
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    new_label = label2id[entities[recipe_idx][word_idx]]
                else:
                    new_label = -100 if only_first_token \
                        else label2id[entities[recipe_idx][word_idx]]
                label_ids.append(new_label)
            previous_word_idx = word_idx
        
        words_beginnings += (max_length - len(words_beginnings)) * [False]
        recipes_words_beginnings.append(words_beginnings)
        if entities:
            labels.append(label_ids)

    if entities:
        tokenized_data["labels"] = labels

    tokenized_data["recipes"] = recipes
    tokenized_data["prediction_mask"] = recipes_words_beginnings

    return tokenized_data


class BertCRFOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    y_preds: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.FloatTensor] = None


class BERTCRF(BertForTokenClassification):
    """
    BERT with CRF layer on top. This class directly follows implementation used
    in "BERTimbau: Pretrained BERT Models for Brazilian Portuguese", which was
    made available under the MIT license and can be found under the following
    link:  https://github.com/neuralmind-ai/portuguese-bert
    """
    def __init__(self, config):
        """
        :param config: BertConfig
        """
        super().__init__(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, prediction_mask=None,):
        
        outputs = {}

        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]
        
        output = self.dropout(output)
        logits = self.classifier(output)

        mask = prediction_mask
        batch_size = logits.shape[0]
        
        outputs['logits'] = logits
        if labels is not None:
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                seq_mask = [i for i, mask in enumerate(seq_mask) if mask]
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels, reduction='token_mean')
            loss /= batch_size
            outputs['loss'] = loss

        else:
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_mask = [i for i, mask in enumerate(seq_mask) if mask]
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                output_tags.append(tags[0])

            output_tags = [tags + [-100] * (128 - len(tags)) for tags in output_tags]
            outputs['predictions'] = torch.tensor(output_tags)
                
        return BertCRFOutput(**outputs)


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

        # to discard pretrained classification layer
        ignore_mismatched_sizes = True if \
            self.config["model_name_or_path"] is not None else False

        if self.config["use_crf"] is True:
            model = BERTCRF.from_pretrained(
                        model_name_or_path,
                        num_labels=len(self.config["label2id"]),
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        label2id=label2id,
                        id2label=id2label,
                        classifier_dropout=0.2
                    )
        else:
            model = BertForTokenClassification.from_pretrained(
                model_name_or_path,
                num_labels=len(self.config["label2id"]),
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                label2id=label2id,
                id2label=id2label,
                classifier_dropout=0.2
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

        if self.config["use_crf"] is True:
            token_labels = preds[0][1]
        else:
            token_probs = preds[0]
            token_labels = token_probs.argmax(axis=2)

        pred_entities = []

        num_of_recipes = dataset.num_rows

        for recipe_idx in range(num_of_recipes):
            text_split_words = recipes[recipe_idx]
            text_split_tokens = self.tokenizer.convert_ids_to_tokens(
                data["input_ids"][recipe_idx])

            id2label = self.trainer.model.config.id2label
            if self.config["use_crf"] is True:  # labels are associated to
                # first subwords, hence, are already the word entities
                word_entities = \
                    [self.trainer.model.config.id2label[word_label] for
                     word_label in token_labels[recipe_idx] if word_label != -100]
            else:
                word_entities = token_to_entity_predictions(
                    text_split_words,
                    text_split_tokens,
                    token_labels[recipe_idx],
                    id2label
                )
            pred_entities.append(word_entities)

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
    CONFIG["use_crf"] = args.use_crf
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
        print(results)
        cross_val_results[fold_id] = results

    with open("bert_cross_val_results.json", "w") as json_file:
        json.dump(cross_val_results, json_file, indent=4)

    # aggregate andprint results
    cross_val_results_aggregated = {
        entity: {"precision": [], "recall": [], "f1": []} for entity in
        ENTITIES + ["all"]
    }

    print(f"{'entity':^20s}{'precision':^15s}{'recall':^15s}{'f1-score':^15s}")
    for entity in cross_val_results_aggregated.keys():
        print(f"{entity:^20s}", end="")
        for metric in cross_val_results_aggregated[entity].keys():
            for fold_id in range(args.num_of_folds):
                cross_val_results_aggregated[entity][metric].append(
                    cross_val_results[fold_id][entity][metric]
                )

            mean = np.mean(cross_val_results_aggregated[entity][metric])
            mean = int(mean * 1000) / 1000
            std = np.std(cross_val_results_aggregated[entity][metric])
            std = int(std * 1000) / 1000 + 0.001 * \
                  round(std - int(std * 1000) / 1000)
            print(f"{mean:^2.3f} +- {std:^2.3f} ", end="")
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
    parser.add_argument("--use-crf", action='store_true',
                        help="Use CRF layer on top of BERT + linear layer")
    args = parser.parse_args()

    cross_validate(args)
