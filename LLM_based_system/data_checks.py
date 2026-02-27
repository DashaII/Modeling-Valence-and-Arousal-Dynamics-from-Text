import json
import ast
from words_classification import split_words_and_essays, get_all_texts
from help_funtions import split_essays
import configs
import re


def check_missing_ids(json_file_name, test_data=False, full_data=False, num_of_buckets=70, from_bucket=0, to_bucket=70):
    _, _, words, essays, user_map = split_words_and_essays(test_data=test_data)
    all_texts = get_all_texts(test_data=test_data)

    if not full_data:
        with open(json_file_name, "r", encoding="utf-8") as fj:
            emo = json.load(fj)
            emo_dict = {}
            for i, em in emo.items():
                emo_dict.update(em)

            buckets = split_essays(n_buckets=num_of_buckets, essays_dict=all_texts, respect_users=False, shuffle=True)
            for i, bucket in enumerate(buckets):
                if from_bucket <= i < to_bucket:
                    for key in bucket:
                        if str(key) not in emo_dict:
                            print(key)
    else:
        with open(json_file_name, "r", encoding="utf-8") as fj:
            data = json.load(fj)
            missing = []
            for idx in all_texts:
                if str(idx) not in data:
                    missing.append(idx)
            print("missing", missing)


def check_naming_consistency(json_file_name, full_data=False):
    wrong_naming = {}
    emotions_to_va_lowered = [emo.lower() for emo in configs.emotions_to_va]
    with open(json_file_name, "r", encoding="utf-8") as fj:
        emo = json.load(fj)
        if not full_data:
            for bucket_num, bucket in emo.items():
                for idx, emotion in bucket.items():
                    if emotion.lower() not in emotions_to_va_lowered:
                        wrong_naming[idx] = emotion
        else:
            for idx, emotion in emo.items():
                if emotion.lower() not in emotions_to_va_lowered:
                    wrong_naming[idx] = emotion
    print("not consistent labels", wrong_naming)


def merge_json_files_with_buckets(merge_to_file_name, list_files):
    with open(merge_to_file_name, "w", encoding="utf-8") as merge_fj:
        merge_data = {}
        for file in list_files:
            with open(file, "r", encoding="utf-8") as fj:
                data = json.load(fj)
                for i, bucket in data.items():
                    merge_data.update(bucket)
        json.dump(merge_data, merge_fj, indent=2, ensure_ascii=False)
    return merge_to_file_name


def merge_json_files_without_buckets(merge_to_file_name, list_files):
    merge_data = {}
    for file in list_files:
        with open(file, "r", encoding="utf-8") as fj:
            data = json.load(fj)
            merge_data.update(data)
    with open(merge_to_file_name, "w", encoding="utf-8") as merge_fj:
        json.dump(merge_data, merge_fj, indent=2, ensure_ascii=False)
    return merge_to_file_name


def merge_json_files_without_buckets_to_existing_file(merge_to_file_name, list_files):
    with open(merge_to_file_name, "r", encoding="utf-8") as merge_fj:
        merge_data = json.load(merge_fj)
        for file in list_files:
            with open(file, "r", encoding="utf-8") as fj:
                data = json.load(fj)
                merge_data.update(data)
    with open(merge_to_file_name, "w", encoding="utf-8") as merge_fj:
        json.dump(merge_data, merge_fj, indent=2, ensure_ascii=False)


def repair_dicts_from_txt(input_file, output_json):
    repaired = {}
    failed_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split "Bucket 46:{...}"
            prefix, dict_str = line.split(":", 1)
            bucket_name = prefix.strip()
            try:
                # parse Python-style dict with single quotes
                parsed = ast.literal_eval(dict_str)
                # Store by bucket name
                repaired[bucket_name] = parsed

            except Exception as e:
                print("Failed line:", line)
                print("Error:", e)
                failed_lines.append(line)
    # Save full repaired output to JSON
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(repaired, f_out, indent=2, ensure_ascii=False)

    return repaired, failed_lines


def from_bad_txt_to_json(txt_path, json_path):
    """
    Reads a TXT file of the form:
    172: {valence: 2, arousal: 0}, 173: {valence: -1, arousal: 0}, ...
    Writes a JSON file of the form:
    {
        "172": {"valence": 2, "arousal": 0},
        ...
    }
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Regex to match entries like: 172: {valence: 2, arousal: 0}
    pattern = re.compile(
        r"(\d+)\s*:\s*\{\s*valence\s*:\s*'?(-?\d+)'?\s*,\s*arousal\s*:\s*'?(-?\d+)'?\s*\}"
    )
    result = {}
    for match in pattern.finditer(text):
        text_id, valence, arousal = match.groups()
        result[text_id] = {
            "valence": int(valence),
            "arousal": int(arousal),
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def merge_json_valence_and_arousal(valence_file, arousal_file, save_to_file):
    with open(valence_file, "r", encoding="utf-8") as vf:
        valence_data = json.load(vf)
    with open(arousal_file, "r", encoding="utf-8") as af:
        arousal_data = json.load(af)

    data_enriched = {}
    for idx in valence_data:
        data_enriched[int(idx)] = {"Valence": valence_data[idx], "Arousal": arousal_data[idx], "Valence-Arousal": (valence_data[idx], arousal_data[idx])}

    with open(save_to_file, "w", encoding="utf-8") as f_enriched:
        json.dump(data_enriched, f_enriched, indent=2, ensure_ascii=False, sort_keys=True)
    return save_to_file


def reformat_json_val_and_aro_together(val_and_aro_file, save_to):
    with open(val_and_aro_file, "r", encoding="utf-8") as vf:
        data = json.load(vf)

    data_enriched = {}
    for idx in data:
        data_enriched[int(idx)] = {"Valence": data[idx]["valence"], "Arousal": data[idx]["arousal"],
                                   "Valence-Arousal": (data[idx]["valence"], data[idx]["arousal"])}

    with open(save_to, "w", encoding="utf-8") as f_enriched:
        json.dump(data_enriched, f_enriched, indent=2, ensure_ascii=False, sort_keys=True)
    return save_to
