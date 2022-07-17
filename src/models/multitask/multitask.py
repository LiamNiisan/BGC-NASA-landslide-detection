import os
import io
import json
import pickle
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

import config
from extraction.time import time
from extraction.casualties import casualties
from extraction.time.landslide_event_time import LandslideEventTime
from extraction.location.landslide_event_location import LandslideEventLocation


CATEGORIES_ID2L = [
    "",
    "complex",
    "creep",
    "debris_flow",
    "earth_flow",
    "lahar",
    "landslide",
    "mudslide",
    "riverbank_collapse",
    "rock_fall",
    "rotational_slide",
    "snow_avalanche",
    "topple",
    "translational_slide",
]

TRIGGERS_ID2L = [
    "",
    "construction",
    "continuous_rain",
    "dam_embankment_collapse",
    "downpour",
    "earthquake",
    "flooding",
    "freeze_thaw",
    "mining",
    "monsoon",
    "rain",
    "snowfall_snowmelt",
    "tropical_cyclone",
    "leaking_pipe",
    "volcano",
]

CATEGORIES_L2ID = {label: i for i, label in enumerate(CATEGORIES_ID2L)}
TRIGGERS_L2ID = {label: i for i, label in enumerate(TRIGGERS_ID2L)}

IOB_ID2L = [
    "O",
    "B-LOC",
    "I-LOC",
    "B-TIME",
    "I-TIME",
    "B-DATE",
    "I-DATE",
    "B-CAS",
    "I-CAS",
]

IOB_L2ID = {tag: i for i, tag in enumerate(IOB_ID2L)}

SPAN_ID2L = [
    "LOC",
    "TIME",
    "DATE",
    "CAS",
]

SPAN_L2ID = {tag: i for i, tag in enumerate(SPAN_ID2L)}


MODEL_NAME = "distilroberta-base"
BATCH_SIZE = 8

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandslideEventsDataset(Dataset):
    """
    Dataset of extracted labels and tags from the NASA landslide dataset.
    """

    def __init__(self, data, is_test=False):
        self.data = data
        self.ids, self.mask = self.get_inputs()

        if not is_test:
            self.categories = self.get_categories()
            self.triggers = self.get_triggers()
            self.spans = self.get_spans()
        else:
            self.categories = torch.zeros(self.ids.shape[0])
            self.triggers = torch.zeros(self.ids.shape[0])
            self.spans = torch.zeros(self.ids.shape[0])

    def get_inputs(self):
        articles = [article["text"] for article in self.data]
        result = TOKENIZER(articles, padding="max_length", truncation=True)
        ids = torch.tensor(result.input_ids)
        mask = torch.tensor(result.attention_mask)
        return ids, mask

    def get_categories(self):
        categories = torch.tensor(
            [CATEGORIES_L2ID[article["EVENT_CATEGORY"]] for article in self.data]
        )
        return categories

    def get_triggers(self):
        triggers = torch.tensor(
            [TRIGGERS_L2ID[article["EVENT_TRIGGER"]] for article in self.data]
        )
        return triggers

    def get_spans(self):
        span_list = []
        for i, article in enumerate(self.data):
            # Initialize all the iobs to O values which are equal to zero
            spans = []
            for label in article["qas"].keys():
                label_value = article["qas"][label].lower()
                if not label_value:
                    spans.append(torch.tensor([0, 0]))
                    continue
                # For loop to match the iob position inside the tokenized input
                found = False
                for j in range(len(self.ids[i])):
                    k = 1
                    # strip and lower to make sure it matches
                    span = TOKENIZER.decode(self.ids[i][j : j + k]).strip().lower()
                    # If the first word matches, we keep adding words
                    # until it's equal or doesn't match anymore
                    while span in label_value:
                        if span == label_value:
                            # If it's equal, we add the iob tags
                            spans.append(torch.tensor([j, j + k]))
                            found = True
                            break
                        k += 1
                        span = TOKENIZER.decode(self.ids[i][j : j + k]).strip().lower()

                    if found:
                        break

                if not found:
                    spans.append(torch.tensor([0, 0]))

            span_list.append(torch.vstack(spans))

        return span_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        categories = self.categories[index]
        triggers = self.triggers[index]
        ids = self.ids[index]
        mask = self.mask[index]
        spans = self.spans[index]
        return ids, categories, triggers, spans, mask


class LandslideEventsSpanClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(LandslideEventsSpanClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.activation = nn.Sigmoid()
        self.start_classifier = nn.Linear(self.hidden_size, 1)
        self.end_classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden_states):
        out_start = self.start_classifier(hidden_states)
        out_start = self.activation(out_start)
        out_end = self.end_classifier(hidden_states)
        out_end = self.activation(out_end)
        return (out_start.squeeze(), out_end.squeeze())


class LandslideEventsLabelClassifier(nn.Module):
    def __init__(self, hidden_size, num_outputs):
        super(LandslideEventsLabelClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_outputs)

    def forward(self, cls_state):
        return self.classifier(cls_state).log_softmax(dim=-1)


class LandslideEventsClassifier(nn.Module):
    def __init__(self):
        super(LandslideEventsClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.category_classifier = LandslideEventsLabelClassifier(
            self.bert.config.hidden_size, len(CATEGORIES_ID2L)
        )
        self.trigger_classifier = LandslideEventsLabelClassifier(
            self.bert.config.hidden_size, len(TRIGGERS_ID2L)
        )
        self.span_classifiers = []
        for qas in SPAN_ID2L:
            self.span_classifiers.append(
                LandslideEventsSpanClassifier(self.bert.config.hidden_size).to(DEVICE)
            )
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.dropout(
            self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        )
        cls_state = hidden_states[:, 0, :]
        categories = self.category_classifier(cls_state)
        triggers = self.trigger_classifier(cls_state)
        spans = []
        for i, qas in enumerate(SPAN_ID2L):
            spans.append(self.span_classifiers[i](hidden_states))

        return categories, triggers, spans


def label_loss_fn(pred_labels, gold_labels, weights):
    cel = nn.CrossEntropyLoss(weight=weights)
    out = cel(pred_labels, gold_labels)
    return out


def span_loss_fn(pred_span, gold_span, mask):
    length = mask.sum()
    cel = nn.CrossEntropyLoss()
    out = cel(pred_span[:, :length], gold_span)
    return out


def pre_train_model(dataloader, model, optimizer):
    model.train()
    total_loss = 0

    for inputs, categories, triggers, spans, masks in tqdm(
        dataloader, total=len(dataloader)
    ):
        model.zero_grad()

        inputs = inputs.to(DEVICE)
        spans = spans.to(DEVICE)
        masks = masks.to(DEVICE)

        pred_categories, pred_triggers, pred_spans = model(
            input_ids=inputs, attention_mask=masks
        )

        span_loss = 0
        for i, s_pred in enumerate([pred_spans[0] + pred_spans[2]]):
            span_loss_start = span_loss_fn(s_pred[0], spans[:, i, 0], masks)
            span_loss_end = span_loss_fn(s_pred[1], spans[:, i, 1], masks)
            span_loss += span_loss_start + span_loss_end

        loss = span_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def train_model(dataloader, model, optimizer, cat_weights, trig_weights):
    model.train()
    total_loss = 0

    for inputs, categories, triggers, spans, masks in tqdm(
        dataloader, total=len(dataloader)
    ):
        model.zero_grad()

        inputs = inputs.to(DEVICE)
        categories = categories.to(DEVICE)
        triggers = triggers.to(DEVICE)
        spans = spans.to(DEVICE)
        masks = masks.to(DEVICE)

        pred_categories, pred_triggers, pred_spans = model(
            input_ids=inputs, attention_mask=masks
        )

        categories_loss = label_loss_fn(pred_categories, categories, cat_weights)
        triggers_loss = label_loss_fn(pred_triggers, triggers, trig_weights)

        span_loss = 0
        for i, s_pred in enumerate(pred_spans):
            span_loss_start = span_loss_fn(s_pred[0], spans[:, i, 0], masks)
            span_loss_end = span_loss_fn(s_pred[1], spans[:, i, 1], masks)
            span_loss += span_loss_start + span_loss_end

        loss = categories_loss + triggers_loss + span_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def eval_model(dataloader, model, report=False, device=DEVICE):
    model.eval()

    categories_gold = []
    categories_pred = []
    triggers_gold = []
    triggers_pred = []
    correct = 0
    span_total = 0

    for inputs, categories, triggers, spans, masks in tqdm(
        dataloader, total=len(dataloader)
    ):
        inputs = inputs.to(device)
        masks = masks.to(device)
        spans = spans.to(device)

        pred_categories, pred_triggers, pred_spans = model(
            input_ids=inputs, attention_mask=masks
        )

        non_zero_label_ids = torch.where(categories != 0)
        categories_gold.extend(categories[non_zero_label_ids])
        categories_pred.extend(pred_categories.argmax(dim=-1)[non_zero_label_ids].cpu())

        non_zero_label_ids = torch.where(categories != 0)
        triggers_gold.extend(triggers[non_zero_label_ids])
        triggers_pred.extend(pred_triggers.argmax(dim=-1)[non_zero_label_ids].cpu())

        for i in range(len(SPAN_ID2L)):
            pred_start = pred_spans[i][0].argmax(dim=-1)
            pred_end = pred_spans[i][1].argmax(dim=-1)

            correct += torch.sum(spans[:, i, 0] == pred_start)
            correct += torch.sum(spans[:, i, 1] == pred_end)
            span_total += inputs.shape[0] * 2

    categories_f1 = f1_score(categories_gold, categories_pred, average="micro")
    triggers_f1 = f1_score(triggers_gold, triggers_pred, average="micro")

    span_accuracy = correct / span_total
    span_accuracy = span_accuracy.item()

    if report:
        print("Classification report for landslide categories")
        print(
            classification_report(
                [CATEGORIES_ID2L[c] for c in categories_gold],
                [CATEGORIES_ID2L[c] for c in categories_pred],
                zero_division=0,
            )
        )
        print("Classification report for landslide triggers")
        print(
            classification_report(
                [TRIGGERS_ID2L[c] for c in triggers_gold],
                [TRIGGERS_ID2L[c] for c in triggers_pred],
                zero_division=0,
            )
        )

    return categories_f1, triggers_f1, span_accuracy


def _select_best_answer_span(start_prob, end_prob, distance):
    best_p = 0.95
    best_span = (0, 0)
    for i, start_p in enumerate(start_prob):
        for y, end_p in enumerate(end_prob[i + 1 : i + distance + 1]):
            p = (start_p + end_p) / 2
            if p > best_p:
                best_p = p
                best_span = (i, i + y + 1)

    return best_span


def select_best_answer_span(start_probs, end_probs, distance, lengths=None):
    spans = []
    if lengths is None:
        lengths = torch.ones(start_probs.shape[0]) * start_probs.shape[1]
    # In case there is only one example to extract
    if len(start_probs.shape) == 1:
        start_probs = start_probs.unsqueeze(dim=0)
        end_probs = end_probs.unsqueeze(dim=0)
        lengths = lengths.unsqueeze(dim=0)
    for start_prob, end_prob, length in zip(start_probs, end_probs, lengths):
        spans.append(
            _select_best_answer_span(
                start_prob[: int(length)], end_prob[: int(length)], distance
            )
        )
    return spans


def predict_batch(model, text, device=DEVICE):
    if type(text) != list:
        text = [text]

    result = TOKENIZER(text, padding="max_length", truncation=True)
    inputs = torch.tensor(result.input_ids)
    masks = torch.tensor(result.attention_mask)
    model.eval()

    inputs = inputs.to(device)
    masks = masks.to(device)

    pred_categories, pred_triggers, pred_spans_logits = model(
        input_ids=inputs, attention_mask=masks
    )

    categories_pred = pred_categories.argmax(dim=-1).cpu()
    triggers_pred = pred_triggers.argmax(dim=-1).cpu()

    lengths = masks.sum(dim=-1)

    pred_spans = []
    for span_id in range(len(SPAN_ID2L)):
        pred_spans.append([])
        spans = select_best_answer_span(
            pred_spans_logits[span_id][0], pred_spans_logits[span_id][1], 5, lengths
        )
        for idx, span in enumerate(spans):
            pred_spans[-1].append(
                TOKENIZER.decode(inputs[idx][span[0] : span[1]]).strip()
            )

    cat_decoded = [CATEGORIES_ID2L[c] for c in categories_pred]
    trigs_decoded = [TRIGGERS_ID2L[c] for c in triggers_pred]

    return cat_decoded, trigs_decoded, pred_spans


def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def predict(article_df):
    """
    Returns all extracted event information predicted from a dataframe of articles
    using BERT on a cpu.

    Parameters
    ----------
    article_df : pandas.DataFrame
        dataframe containing all the articles

    Returns
    -------
    dict
        dictionary of lists with all the predictions.
    """    
    categories = []
    triggers = []
    extractions = []

    articles = article_df["article_text"].to_numpy().tolist()
    publication_dates = article_df["article_publish_date"].astype(str).to_numpy()
    publication_dates = list(map(time.str_to_datetime, publication_dates))

    with open(
        os.path.join(config.model_path, "landslide_detection-QA-2-epoch_2-40.model"), "rb"
    ) as f:
        model = CPU_Unpickler(f).load()

    with torch.no_grad():
        for i in range(len(SPAN_ID2L)):
            extractions.append([])
        config.logger.info("running multitask model...")
        for batch in tqdm(
            get_batch(articles, BATCH_SIZE), total=round(len(articles) / BATCH_SIZE)
        ):
            batch_categories, batch_triggers, batch_extractions = predict_batch(
                model, batch, "cpu"
            )
            categories.extend(batch_categories)
            triggers.extend(batch_triggers)
            for i, extracted_entities in enumerate(batch_extractions):
                extractions[i].extend(extracted_entities)

    extracted_locations = extractions[SPAN_L2ID["LOC"]]
    extracted_times = extractions[SPAN_L2ID["TIME"]]
    extracted_dates = extractions[SPAN_L2ID["DATE"]]
    event_casualties = [casualties.format_num(n) for n in extractions[SPAN_L2ID["CAS"]]]

    event_locations = Parallel(n_jobs=-1, verbose=1)(
        delayed(LandslideEventLocation)([location]) for location in extracted_locations
    )

    event_times = []
    for i in range(len(articles)):
        time_phrases = [extracted_dates[i], extracted_times[i]]
        publication_date = publication_dates[i]
        event_times.append(LandslideEventTime(time_phrases, publication_date))

    return {
        "location": event_locations,
        "time": event_times,
        "casualties": event_casualties,
        "category": categories,
        "trigger": triggers,
    }