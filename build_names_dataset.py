import csv
import json
import random
import re


non_alphabetic = re.compile(r"[^a-zA-Z-]")


def build_names_dataset():
    """ Build a dataset of single tokens that are either names or not names. """
    non_name_words = load_english_words()
    surnames = load_surnames()
    baby_names = load_baby_names()

    rows = [
        {
            "tokens": [word],
            "ner_tags": [0],
            "is_name": False,
            "name_type": None,
        }
        for word in non_name_words
    ]
    rows += [
        {
            "tokens": [surname],
            "ner_tags": [1],
            "is_name": True,
            "name_type": "last",
        }
        for surname in surnames
    ]
    rows += [
        {
            "tokens": [name],
            "ner_tags": [1],
            "is_name": True,
            "name_type": "first",
        }
        for name in baby_names
    ]

    # TODO: Create more full names
    full_names = zip(baby_names, surnames[:len(baby_names)])
    rows += [
        {
            "tokens": [first, last],
            "ner_tags": [1, 2],
            "is_name": True,
            "name_type": "full",
        }
        for first, last in full_names
    ]

    random.seed(1234)
    random.shuffle(rows)
    rows = [
        {"id": i} | row
        for i, row in enumerate(rows)
    ]

    with open("data/names_dataset.csv", "w") as f:
        csvwriter = csv.DictWriter(f, fieldnames=rows[0].keys())
        csvwriter.writeheader()
        csvwriter.writerows(rows)

    print(f"Wrote {len(rows)} rows to data/names_dataset.csv.")


def load_english_words() -> list[str]:
    non_name_words = []
    with open("data/sources/kaggle_english_words/words.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line.istitle():
                # Exclude proper nouns just to be careful. Some may be names
                continue
            if bool(non_alphabetic.search(line)):
                # Exclude words with non-alphabetic characters (except hyphens)
                continue
            non_name_words.append(line.lower())

    non_name_words = list(set(non_name_words))
    print(f"Loaded {len(non_name_words)} English words.")
    return non_name_words


def load_surnames() -> list[str]:
    surnames = []
    with open("data/sources/fivethirtyeight_names/surnames.csv", "r") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            surnames.append(row["name"].lower())

    surnames = list(set(surnames))
    print(f"Loaded {len(surnames)} surnames.")
    return surnames


def load_baby_names() -> list[str]:
    national_names = []
    with open("data/sources/kaggle_baby_names/NationalNames.csv", "r") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            national_names.append(row["Name"].lower())

    state_names = []
    with open("data/sources/kaggle_baby_names/StateNames.csv", "r") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            state_names.append(row["Name"].lower())

    baby_names = list(set(national_names) | set(state_names))
    print(f"Loaded {len(baby_names)} baby names.")
    return baby_names


if __name__ == "__main__":
    build_names_dataset()
