import re
import pandas as pd
import configs
from ask_ollama import ask_ollama_chat
import json
import ast
import copy
from ask_openai import ask_openai_chat


def get_texts_split_by_user():
    semeval_data = pd.read_csv(configs.test_data_emo)
    # Convert timestamp to datetime for proper sorting
    semeval_data["timestamp"] = pd.to_datetime(semeval_data["timestamp"])

    # Sort globally by: user_id > timestamp
    semeval_data = semeval_data.sort_values(
        by=["user_id", "timestamp"],
        ascending=True
    )
    train_semeval_data = semeval_data[semeval_data["train_data"] == True]
    test_semeval_data_seen = semeval_data[
        (semeval_data["train_data"] == False) &
        (semeval_data["is_seen_user"] == True)
        ]
    test_semeval_data_unseen = semeval_data[
        (semeval_data["train_data"] == False) &
        (semeval_data["is_seen_user"] == False)
        ]

    test_texts_dict = {}
    for _, row in test_semeval_data_seen.iterrows():  # iterate row by row
        user_id = row["user_id"]
        phase = row["collection_phase"]
        text_id = row["text_id"]

        # Initialize nested dictionaries
        if user_id not in test_texts_dict:
            test_texts_dict[user_id] = {}
        if phase not in test_texts_dict[user_id]:
            test_texts_dict[user_id][phase] = {}

        test_texts_dict[user_id][phase][text_id] = {
            "text": row["text"],
            "timestamp": str(row["timestamp"]),
            "is_words": row["is_words"]
        }

    train_texts_dict = {}
    for _, row in train_semeval_data.iterrows():  # iterate row by row
        user_id = row["user_id"]
        text_id = row["text_id"]

        # Initialize nested dictionaries
        if user_id not in train_texts_dict:
            train_texts_dict[user_id] = {}

        train_texts_dict[user_id][text_id] = {
            "text": row["text"],
            "timestamp": str(row["timestamp"]),
            "is_words": row["is_words"],
            "valence": row["valence"],
            "arousal": row["arousal"],
            "emotion": row["emotion"],
        }

    return test_texts_dict, train_texts_dict


def split_by_train_predict(train_len = 20):
    test_data, train_data = get_texts_split_by_user()

    train_text_ids, test_text_ids = [], []
    for user_id in test_data:
        for phase in test_data[user_id]:
            train_texts, test_texts = {}, {}
            # for i, text_id in enumerate(test_texts[user_id][phase]):
            test_texts = copy.deepcopy(test_data[user_id][phase])
            test_text_ids.extend(test_texts.keys())

            max_train_len = len(train_data[user_id])
            if max_train_len > train_len:
                for i, text_id in enumerate(train_data[user_id]):
                    if max_train_len - train_len - i <= 0:
                        train_texts[text_id] = train_data[user_id][text_id]
            else:
                train_texts = train_data[user_id]

            test_data[user_id][phase]["train_texts"] = train_texts
            test_data[user_id][phase]["test_texts"] = test_texts
    # print(test_text_ids)
    return test_data, train_text_ids, test_text_ids


def run_user_aware_prompt_static(save_to_file_name, model_name = configs.model_gpt_oss_120b, openai = False, prompt_type="emotion", train_len = 20, from_b=0, to_b=200):
    all_texts = split_by_train_predict(train_len = train_len)[0]

    # two output files: json for valid dicts and txt for non-parseable dicts
    json_file = save_to_file_name + ".json"
    bad_file = save_to_file_name + "_BAD.txt"

    count = 0
    with open(json_file, "w", encoding="utf-8") as fj, open(bad_file, "w", encoding="utf-8") as fb:
        all_results = {}  # all valid parsed dicts
        for user_id in all_texts:
            for phase in all_texts[user_id]:
                train_texts = []
                prediction_texts = {}
                for text_id in all_texts[user_id][phase]["train_texts"]:
                    text = all_texts[user_id][phase]["train_texts"][text_id]["text"]

                    if prompt_type == "emotion":
                        label = all_texts[user_id][phase]["train_texts"][text_id]["emotion"]
                        train_texts.append(text + " → " + label)
                for text_id in all_texts[user_id][phase]["test_texts"]:
                    text = all_texts[user_id][phase]["test_texts"][text_id]["text"]
                    prediction_texts[text_id] = text

                if prompt_type == "emotion":
                    prompt = configs.prompt_user_aware_static_emotion
                enriched_prompt = prompt.format(train=train_texts, predict=prediction_texts)

                count += 1
                if from_b <= count < to_b:
                    print("prompt count", count)
                    if openai:
                        raw_model_response = ask_openai_chat(prompt=enriched_prompt, model_name=model_name)
                    else:
                        raw_model_response = ask_ollama_chat(prompt=enriched_prompt, model_name=model_name)

                    # --- sanitize before attempting json parsing ---
                    sanitized = sanitize_json_like(raw_model_response)

                    # sanitize_json_like() succeeded and returns a dict
                    if isinstance(sanitized, dict):
                        all_results.update(sanitized)
                        continue
                    else:
                        fb.write(f"{count}:" + raw_model_response + "\n")
        # after loop, save all valid JSON results at once
        json.dump(all_results, fj, indent=2, ensure_ascii=False)
        print("Saved good JSON:", json_file)
        print("Saved bad responses:", bad_file)
    return json_file


def sanitize_json_like(text):
    # fix numeric keys
    t = re.sub(r'{(\s*)(\d+)(\s*):', r'{"\2":', text)
    t = re.sub(r',(\s*)(\d+)(\s*):', r',"\2":', t)
    # try JSON
    try:
        return json.loads(t)
    except:
        pass
    # try Python dict
    try:
        data = ast.literal_eval(t)
        return data
    except:
        return None
