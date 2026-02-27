import pandas as pd
import json
import configs
from typing import Dict, List, Tuple
import random
from words_classification import split_words_and_essays, get_all_texts
import csv
import zipfile
import os

# split essays into buckets of similar lengths so that one user is not split between two buckets
def split_essays(
    n_buckets: int,
    essays_dict: Dict[int, str],
    user_map: Dict[int, int] = None,
    respect_users: bool = True,
    shuffle = True
) -> List[Dict[int, str]]:

    # If we do NOT want to respect users, we simply split essays independently
    if not respect_users:
        # Convert essays into list of (id, text)
        items = list(essays_dict.items())

        if shuffle:
            random.seed(42)
            random.shuffle(items)
        else:
            # Sort by length descending
            items.sort(key=lambda x: len(x[1]), reverse=True)

        buckets = [{} for _ in range(n_buckets)]
        bucket_lengths = [0] * n_buckets

        for essay_id, text in items:
            min_idx = bucket_lengths.index(min(bucket_lengths))
            buckets[min_idx][essay_id] = text
            bucket_lengths[min_idx] += len(text)
        return buckets

    # group essays by user
    user_to_essays: Dict[int, List[Tuple[int, str]]] = {}
    user_total_length: Dict[int, int] = {}
    for essay_id, text in essays_dict.items():
        user = user_map[essay_id]
        user_to_essays.setdefault(user, []).append((essay_id, text))
        user_total_length[user] = user_total_length.get(user, 0) + len(text)

    # sort users by total length descending
    sorted_users = sorted(user_total_length.items(), key=lambda x: x[1], reverse=True)

    # initialize empty buckets
    buckets: List[Dict[int, str]] = [{} for _ in range(n_buckets)]
    bucket_lengths = [0] * n_buckets

    # assign each user block to the bucket with the smallest total length
    for user, _ in sorted_users:
        # Find bucket with the smallest total length
        min_idx = bucket_lengths.index(min(bucket_lengths))
        for essay_id, text in user_to_essays[user]:
            buckets[min_idx][essay_id] = text
            bucket_lengths[min_idx] += len(text)
    return buckets


# split essays and words SEPARATELY into buckets of similar lengths
def split_essays_words(n_buckets_essays: int, n_buckets_words: int, shuffle = True):
    """
    :param n_buckets_essays: split essays into n buckets
    :param n_buckets_words: split words into n buckets
    :param shuffle: default True
    :return: essay_buckets, words_buckets
    USEFUL SPLIT: 14 for words and 84 for essays
    """
    def get_buckets(items, n, shuffle = True):
        if shuffle:
            random.seed(42)
            random.shuffle(items)
        else:
            # Sort by length descending
            items.sort(key=lambda x: len(x[1]), reverse=True)

        buckets = [{} for _ in range(n)]
        bucket_lengths = [0] * n

        for essay_id, text in items:
            min_idx = bucket_lengths.index(min(bucket_lengths))
            buckets[min_idx][essay_id] = text
            bucket_lengths[min_idx] += len(text)
        print("bucket len", bucket_lengths[0])
        return buckets

    _, _, words, essays, user_map = split_words_and_essays()

    # Convert essays and words into list of (id, text)
    items_essays = list(essays.items())
    items_words = [(k, " , ".join(v)) for k,v in words.items()]

    essays_buckets = get_buckets(items_essays, n_buckets_essays)
    words_buckets = get_buckets(items_words, n_buckets_words)

    return essays_buckets, words_buckets


def add_emotion_to_csv(from_file, to_file):
    # Function to map each row
    def map_emotion(row):
        key = (row["valence"], row["arousal"])
        return configs.va_to_emotions.get(key, "UNKNOWN")

    df = pd.read_csv(from_file)
    df["emotion"] = df.apply(map_emotion, axis=1)
    # save updated CSV
    df.to_csv(to_file, index=False)

    print(df.head())


def add_va_to_emotion_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    emotions_to_va_lowered = {emo.lower():configs.emotions_to_va[emo] for emo in configs.emotions_to_va}

    data_enriched = {}
    for idx, emotion in data.items():
        va = emotions_to_va_lowered[emotion.lower()]
        data_enriched[int(idx)] = {"Emotion": emotion.lower(), "Valence": va[0], "Arousal": va[1], "Valence-Arousal": va}

    file_name_enriched = file_name.rsplit(".json", 1)[0] + "_va.json"
    with open(file_name_enriched, "w", encoding="utf-8") as f_enriched:
        json.dump(data_enriched, f_enriched, indent=2, ensure_ascii=False, sort_keys=True)
    return file_name_enriched


# takes final json with results and generates .csv with user_id,text_id,pred_valence,pred_arousal columns
def generate_full_csv_for_submission(in_file_name, out_file_name, test_data=True, zip_output=True, zip_name="submission.zip"):
    """
    Generates submission CSV with columns:
    user_id, text_id, pred_valence, pred_arousal
    """
    with open(in_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    emotions_to_va_lowered = {emo.lower(): configs.emotions_to_va[emo] for emo in configs.emotions_to_va}

    pred_map = {}
    for text_id, emotion in data.items():
        va = emotions_to_va_lowered[emotion.lower()]
        pred_map[int(text_id)] = (va[0], va[1])

    if test_data:
        file_csv = configs.raw_test_data
    else:
        file_csv = configs.raw_train_data

    rows_out = []
    with open(file_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_id = int(row["text_id"])
            user_id = row["user_id"]

            if text_id not in pred_map:
                raise ValueError(f"Missing prediction for text_id={text_id}")

            valence, arousal = pred_map[text_id]

            rows_out.append({
                "user_id": user_id,
                "text_id": text_id,
                "pred_valence": valence,
                "pred_arousal": arousal
            })

    # ---------- Write output CSV ----------
    with open(out_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "text_id", "pred_valence", "pred_arousal"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    # ---------- Optional ZIP ----------
    if zip_output:
        csv_dir = os.path.dirname(configs.raw_test_data)
        zip_path = os.path.join(csv_dir, zip_name)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(
                configs.raw_test_data,
                arcname=os.path.basename(configs.raw_test_data)
            )


def generate_partial_csv_for_submission(
    json_file_name,
    emotion = False,
    test_data=True,
    zip_output=True,
    zip_name="submission.zip",
):
    """
    Works with partial json_file_name.
    Generates:
    1) prediction CSV:
       user_id, text_id, pred_valence, pred_arousal
    2) gold CSV:
       user_id, text_id, gold_valence, gold_arousal
       (gold taken from raw data)
    3) template CSV:
       user_id, text_id, pred_valence[empty], pred_arousal[empty]
    Both CSVs contain ONLY text_ids present in json_file_name.
    """
    # ---------- Load predictions JSON ----------
    with open(json_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize text_ids to int
    pred_text_ids = {int(tid) for tid in data.keys()}
    # Map emotion -> (valence, arousal)
    emotions_to_va_lowered = {emo.lower(): configs.emotions_to_va[emo] for emo in configs.emotions_to_va}

    pred_map = {}
    if emotion:
        for text_id, emotion in data.items():
            va = emotions_to_va_lowered[emotion.lower()]
            pred_map[int(text_id)] = (va[0], va[1])
    else:
        for text_id, va_dict in data.items():
            pred_map[int(text_id)] = (va_dict["Valence"], va_dict["Arousal"])

    # ---------- Select raw file ----------
    raw_csv = configs.raw_test_data if test_data else configs.raw_train_data

    # ---------- Output paths ----------
    base_name = os.path.splitext(os.path.basename(json_file_name))[0]
    base_out_dir = os.path.dirname(json_file_name) or "."
    out_dir = os.path.join(base_out_dir, "submission_eval")
    # create folder if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    pred_csv_path = os.path.join(out_dir, f"{base_name}_pred.csv")
    gold_csv_path = os.path.join(out_dir, f"{base_name}_gold.csv")
    template_csv_path = os.path.join(out_dir, f"{base_name}_template.csv")

    pred_rows = []
    gold_rows = []
    template_rows = []

    # ---------- Read raw CSV and filter ----------
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_id = int(row["text_id"])

            if text_id not in pred_text_ids:
                continue  # ignore text_ids not in JSON

            user_id = row["user_id"]

            # Prediction row
            pv, pa = pred_map[text_id]
            pred_rows.append({
                "user_id": user_id,
                "text_id": text_id,
                "pred_valence": pv,
                "pred_arousal": pa
            })

            # Gold row (from raw data)
            gold_rows.append({
                "user_id": user_id,
                "text_id": text_id,
                "valence": row["valence"],
                "arousal": row["arousal"]
            })

            # Template row (empty)
            template_rows.append({
                "user_id": user_id,
                "text_id": text_id,
                "pred_valence": "",
                "pred_arousal": ""
            })

    # ---------- Write prediction CSV ----------
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "text_id", "pred_valence", "pred_arousal"]
        )
        writer.writeheader()
        writer.writerows(pred_rows)

    # ---------- Write gold CSV ----------
    with open(gold_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "text_id", "valence", "arousal"]
        )
        writer.writeheader()
        writer.writerows(gold_rows)

    # ---------- Write template CSV ----------
    with open(template_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "text_id", "pred_valence", "pred_arousal"]
        )
        writer.writeheader()
        writer.writerows(template_rows)

    # ---------- Optional ZIP ----------
    if zip_output:
        zip_path = os.path.join(out_dir, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_csv_path, arcname=os.path.basename(pred_csv_path))
            zf.write(gold_csv_path, arcname=os.path.basename(gold_csv_path))
            zf.write(template_csv_path, arcname=os.path.basename(template_csv_path))

    return pred_csv_path, gold_csv_path, template_csv_path
