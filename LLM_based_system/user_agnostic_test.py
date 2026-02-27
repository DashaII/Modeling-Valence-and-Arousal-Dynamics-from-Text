from help_funtions import split_essays
from words_classification import split_words_and_essays, get_all_texts
import configs
from ask_ollama import ask_ollama, ask_ollama_chat
from ask_openai import ask_openai_chat
import json
import re
import pandas as pd
from collections import defaultdict


def run_user_agnostic_prompt(prompt, model_name, save_to_file_name, test_data=False, openai=False, shuffled=True, length_mode="text_len",  num_of_buckets=70, from_bucket=0, to_bucket=70):
    # split all texts by buckets
    if shuffled:
        all_texts = get_all_texts(test_data=test_data)
        buckets = split_essays(n_buckets=num_of_buckets, essays_dict=all_texts, respect_users=False)
    else:
        buckets = build_not_shuffled_buckets(num_buckets=num_of_buckets, test_data=test_data, length_mode=length_mode)

    # two output files: json for valid dicts and txt for non-parseable dicts
    json_file = save_to_file_name + ".json"
    bad_file = save_to_file_name + "_BAD.txt"

    # open file once in append mode
    with open(json_file, "w", encoding="utf-8") as fj, open(bad_file, "w", encoding="utf-8") as fb:
        all_results = {}  # all valid parsed dicts

        # for each bucket run prompt
        for i, bucket in enumerate(buckets):
            print("prompt count", i+1)
            if from_bucket <= i < to_bucket:
                if openai:
                    model_response = ask_openai_chat(prompt = prompt.format(bucket), model_name=model_name)
                else:
                    model_response = ask_ollama_chat(prompt=prompt.format(bucket), model_name=model_name)
                # --- TRY PARSING AS JSON ---
                # fix unquoted numeric keys:   {123: "x"} → {"123": "x"}
                model_response = re.sub(r'{(\s*)(\d+)(\s*):', r'{"\2":', model_response)
                model_response = re.sub(r',(\s*)(\d+)(\s*):', r',"\2":', model_response)
                try:
                    response_dict = json.loads(model_response)
                    # response is valid JSON -> store it
                    all_results[i] = response_dict
                except json.JSONDecodeError:
                    # invalid JSON -> save raw text for inspection
                    fb.write(f"Bucket {i + 1}:" + model_response + "\n")
        # after loop, save all valid JSON results at once
        json.dump(all_results, fj, indent=2, ensure_ascii=False)
        print("Saved good JSON:", json_file)
        print("Saved bad responses:", bad_file)
    return json_file


def run_user_agnostic_prompt_subset(prompt, save_to_file_name, subset_idx, model_name):
    _, _, words, essays, user_map = split_words_and_essays()
    all_texts = get_all_texts()
    subset_idx = set(subset_idx)
    text_subset = {k: all_texts[k] for k in subset_idx if k in all_texts}

    # two output files: json for valid dicts and txt for non-parseable dicts
    json_file = save_to_file_name + ".json"
    bad_file = save_to_file_name + "_BAD.txt"

    # open file once in append mode
    with open(json_file, "w", encoding="utf-8") as fj, open(bad_file, "w", encoding="utf-8") as fb:
        model_response = ask_ollama_chat(prompt.format(text_subset), model_name=model_name)

        # --- TRY PARSING AS JSON ---
        # fix unquoted numeric keys:   {123: "x"} → {"123": "x"}
        model_response = re.sub(r'{(\s*)(\d+)(\s*):', r'{"\2":', model_response)
        model_response = re.sub(r',(\s*)(\d+)(\s*):', r',"\2":', model_response)
        try:
            response_dict = json.loads(model_response)
        except json.JSONDecodeError:
            # invalid JSON -> save raw text for inspection
            fb.write(f"{model_response}" + "\n")

        json.dump(response_dict, fj, indent=2, ensure_ascii=False)
        print("Saved good JSON:", json_file)
        print("Saved bad responses:", bad_file)


def build_not_shuffled_buckets(num_buckets: int, test_data=False, length_mode="text_len"):
    """
    Returns:
        result_buckets: list[dict[text_id, text]]
            result_buckets[bucket_idx][text_id] = text
    """
    if test_data:
        df = pd.read_csv(configs.raw_test_data)
    else:
        df = pd.read_csv(configs.raw_train_data)
    # Ensure correct ordering
    df = df.sort_values(by=["user_id", "timestamp"], ascending=True).reset_index(drop=True)
    # Prepare length signal
    if length_mode == "text_len":
        df["item_len"] = df["text"].astype(str).str.len()
        total_len = df["item_len"].sum()
    elif length_mode == "count":
        df["item_len"] = 1
        total_len = len(df)

    target_bucket_len = total_len / num_buckets

    buckets = defaultdict(dict)
    bucket_idx = 0
    current_bucket_len = 0

    # Group by user (keeps sorted order)
    for user_id, user_df in df.groupby("user_id", sort=False):

        for _, row in user_df.iterrows():
            # Move to next bucket if target reached
            if (
                    current_bucket_len >= target_bucket_len
                    and bucket_idx < num_buckets - 1
            ):
                bucket_idx += 1
                current_bucket_len = 0

            buckets[bucket_idx][row["text_id"]] = row["text"]
            current_bucket_len += row["item_len"]
    # Return as ordered list
    result_buckets = [buckets[i] for i in range(len(buckets))]

    return result_buckets


def check_missing_ids(json_file_name, num_of_buckets=70, from_bucket=0, to_bucket=70):
    with open(json_file_name, "r", encoding="utf-8") as fj:
        emo = json.load(fj)
        emo_dict = {}
        for i, em in emo.items():
            emo_dict.update(em)

        _, _, words, essays, user_map = split_words_and_essays()
        all_texts = get_all_texts()
        buckets = split_essays(n_buckets=num_of_buckets, essays_dict=all_texts, respect_users=False, shuffle=True)
        for i, bucket in enumerate(buckets):
            if from_bucket <= i < to_bucket:
                for key in bucket:
                    if str(key) not in emo_dict:
                        print(key)


def check_naming_consistency(json_file_name):
    with open(json_file_name, "r", encoding="utf-8") as fj:
        emo = json.load(fj)
        for bucket_num, bucket in emo.items():
            for idx, emotion in bucket.items():
                if emotion not in configs.emotions_to_va:
                    print(idx, emotion)


def merge_json_files(merge_to_file_name, list_files):
    with open(merge_to_file_name, "w", encoding="utf-8") as merge_fj:
        merge_data = {}
        for file in list_files:
            with open(file, "r", encoding="utf-8") as fj:
                data = json.load(fj)
                for i, bucket in data.items():
                    merge_data.update(bucket)
        json.dump(merge_data, merge_fj, indent=2, ensure_ascii=False)
