import pandas as pd
from collections import Counter
import csv
import configs

def get_all_texts(test_data=False, exclude_seen_users=True):
    if test_data:
        semeval_data = pd.read_csv(configs.raw_test_data)
        if exclude_seen_users:
            semeval_data = semeval_data[semeval_data["is_seen_user"] == False]
            semeval_data = semeval_data.reset_index(drop=True)
    else:
        semeval_data = pd.read_csv(configs.raw_train_data)
    all_texts_dict = dict(zip(semeval_data.text_id, semeval_data.text))
    return all_texts_dict


def split_words_and_essays(test_data=False, stats=False, limit=70):
    if test_data:
        semeval_data = pd.read_csv(configs.raw_test_data)
    else:
        semeval_data = pd.read_csv(configs.raw_train_data)
    words_df = semeval_data[semeval_data.is_words == True]

    words = words_df.text.tolist()
    avg_len = sum(map(len, words)) / len(words) if words else 0
    if stats:
        print(f"Average length: {avg_len:.2f}") # consider inputs longer than 60 to be "is_words = False"

    # feeling words
    words_df_short = words_df[words_df.text.str.len() <= limit]
    words = words_df_short.text.tolist()
    words_idxs = words_df_short.text_id.tolist()

    all_words = []
    all_words_dict = {}
    for i, text in enumerate(words):
        split = text.lower().split(' , ')
        all_words.extend(split)
        all_words_dict[words_idxs[i]] = split

    unique_words = set(all_words)
    if stats:
        print("all_words", len(all_words))
        print("unique_words", len(unique_words))

    # essays
    essays_df = pd.concat([
        semeval_data[~semeval_data.is_words],
        words_df[words_df.text.str.len() > limit]
    ])
    all_essays_dict = dict(zip(essays_df.text_id, essays_df.text))
    user_map_essays = dict(zip(essays_df.text_id, essays_df.user_id))
    user_map_words = dict(zip(words_df.text_id, words_df.user_id))
    user_map = user_map_essays | user_map_words

    if stats:
        print("all_essays_dict len", len(all_essays_dict))
        print("all_words_dict len", len(all_words_dict))

    return unique_words, all_words, all_words_dict, all_essays_dict, user_map


def most_common_feeling_words():
    all_words = split_words_and_essays()[1]
    counter = Counter(all_words)
    # print(counter)

    most_common_counter = 0
    header = ['id', 'word', 'counter', 'acc_counter']
    with open('data/feeling_words.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (key, value) in enumerate(counter.most_common()):
            most_common_counter += value
            # print(i, key, value, "counter", most_common_counter)
            writer.writerow([i, key, value, most_common_counter])

    return counter.most_common()
