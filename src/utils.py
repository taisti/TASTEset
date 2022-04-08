import pandas as pd
import os
import json
import spacy
from spacy.training import biluo_tags_to_offsets, offsets_to_biluo_tags
from nervaluate import Evaluator


NLP = spacy.load('en_core_web_sm')
ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS",
            "PHYSICAL_QUALITY", "PROCESS", "PURPOSE", "PART"]


def prepare_data(taste_set, entities_format="spans"):
    """
    :param tasteset: TASTEset as pd.DataFrame or a path to the TASTEset
    :param entities_format: the format of entities. If equal to 'bio', entities
    will be of the following format: [[B-FOOD, I-FOOD, O, ...], [B-UNIT, ...]].
    If equal to span, entities will be of the following format:
    [[(0, 6, FOOD), (10, 15, PROCESS), ...], [(0, 2, UNIT), ...]]
    :return: list of recipes and corresponding list of entities
    """

    assert entities_format in ["bio", "spans"],\
        'You provided incorrect entities format!'
    if isinstance(taste_set, pd.DataFrame):
        df = taste_set
    elif isinstance(taste_set, str) and os.path.exists(taste_set):
        df = pd.read_csv(taste_set)
    else:
        raise ValueError('Incorret TASTEset format!')

    all_recipes = df["ingredients"].to_list()
    all_entities = []

    for idx in df.index:
        ingredients_entities = json.loads(df.at[idx, "ingredients_entities"])
        entities = []

        for entity_dict in ingredients_entities:
            entities.append((entity_dict["start"], entity_dict["end"],
                             entity_dict["type"]))

        if entities_format == "bio":
            tokenized_recipe, entities = span_to_bio(all_recipes[idx], entities)
            all_recipes[idx] = tokenized_recipe

        all_entities.append(entities)

    return all_recipes, all_entities


def bio_to_biluo(bio_entities):
    """
    :param bio_entities: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD",
    "B-PROCESS"]
    :return: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD", "U-PROCESS"]
    """
    biluo_entities = []

    for entity_idx in range(len(bio_entities)):
        cur_entity = bio_entities[entity_idx]
        next_entity = bio_entities[entity_idx + 1] if \
            entity_idx < len(bio_entities) - 1 else ""

        if cur_entity.startswith("B-"):
            if next_entity.startswith("I-"):
                biluo_entities.append(cur_entity)
            else:
                biluo_entities.append(re.sub("B-", "U-", cur_entity))
        elif cur_entity.startswith("I-"):
            if next_entity.startswith("I-"):
                biluo_entities.append(cur_entity)
            else:
                biluo_entities.append(re.sub("I-", "L-", cur_entity))
        else:  # O
            biluo_entities.append(cur_entity)

    return biluo_entities


def biluo_to_span(recipe, biluo_entities):
    """
    :param biluo_entities: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"]
    :return: list of span entities, eg. [(span_start, span_end, "FOOD"),
    (span_start, span_end, "PROCESS")]
    """
    doc = NLP(recipe)
    spans = biluo_tags_to_offsets(doc, biluo_entities)
    return spans


def bio_to_span(recipe, bio_entities):
    """
    :param bio_entities: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD",
    "B-PROCESS"]
    :return: list of span entities, eg. [(span_start, span_end, "FOOD"),
    (span_start, span_end, "PROCESS")]
    """
    biluo_entities = bio_to_biluo(bio_entities)
    spans = biluo_to_span(recipe, biluo_entities)
    return spans


def span_to_biluo(recipe, span_entities):
    """
    :param span_entities: list of span entities, eg. [(span_start, span_end,
    "FOOD"), (span_start, span_end, "PROCESS")]
    :return: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"] along with tokenized recipe
    """
    doc = NLP(recipe.replace("\n", " "))
    tokenized_recipe = [token.text for token in doc]
    spans = offsets_to_biluo_tags(doc, span_entities)
    return tokenized_recipe, spans


def biluo_to_bio(biluo_entities):
    """
    :param biluo_entities: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"]
    :return: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD", "B-PROCESS"]
    """
    bio_entities = [entity.replace("L-", "I-").replace("U-", "B-")
                    for entity in biluo_entities]
    return bio_entities


def span_to_bio(recipe, span_entities):
    """
    :param span_entities: list of span entities, eg. [(span_start, span_end,
    "FOOD"), (span_start, span_end, "PROCESS")]
    :return: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD", "B-PROCESS"]
    """
    tokenized_recipe, biluo_entities = span_to_biluo(recipe, span_entities)
    bio_entities = biluo_to_bio(biluo_entities)
    return tokenized_recipe, bio_entities


def spans_to_prodigy_spans(list_of_spans):
    """
    Convert to spans format required by nerevaluate.
    """
    prodigy_list_of_spans = []
    for spans in list_of_spans:
        prodigy_spans = []
        for start, end, entity in spans:
            prodigy_spans.append({"label": entity, "start": start, "end": end})
        prodigy_list_of_spans.append(prodigy_spans)

    return prodigy_list_of_spans


def evaluate_predictions(true_entities, pred_entities, entities_format):
    """
    :param true_entities: list of true entities
    :param pred_entities: list of predicted entities
    :param format: format of provided entities. If equal to 'bio', entities
    are expected of the following format: [[B-FOOD, I-FOOD, O, ...],
    [B-UNIT, ...]]. If equal to span, entities are expected of the following
    format: [[(0, 6, FOOD), (10, 15, PROCESS), ...], [(0, 2, UNIT), ...]]
    :return: metrics for the predicted entities
    """

    assert entities_format in ["bio", "spans"],\
        'You provided incorrect entities format!'

    if entities_format == "spans":
        true_entities = spans_to_prodigy_spans(true_entities)
        pred_entities = spans_to_prodigy_spans(pred_entities)

        evaluator = Evaluator(true_entities, pred_entities, tags=ENTITIES)
    else:
        evaluator = Evaluator(true_entities, pred_entities, tags=ENTITIES,
                              loader="list")

    results, results_per_tag = evaluator.evaluate()

    results = results["strict"]

    for entity in results_per_tag.keys():
        results_per_tag[entity] = results_per_tag[entity]["strict"]

    return results, results_per_tag
