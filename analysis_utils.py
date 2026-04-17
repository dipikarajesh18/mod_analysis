import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import html


def load_mod_data_per_game(game_name):
    path = Path(f"data/mods/{game_name}_mods.json")
    df = pd.read_json(path)
    return df

def combine_all_games_data(game_df_list):
    all_dfs = []
    for game_name, df in game_df_list.items():
        df["game"] = game_name
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def plot_value_distribution(df, category_values, category_column="modCategory", metric="Frequency", game_name="All 3 Games"):
    category_values.head(10).plot(kind="bar")
    plt.title(f"Distribution of {category_column} for {game_name}")
    plt.xlabel(category_column)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

def get_category_values(df, game_name="", category_column="modCategory", type="count"):
    if game_name and "game" in df.columns:
        df = df[df["game"] == game_name]
    category_percent = (
        df[category_column]
        .value_counts(normalize=True)
        .mul(100)
    )
    category_count = df[category_column].value_counts()
    if type == "count":
        return category_count
    elif type == "percent":
        return category_percent

def get_downloads_per_category(df, game_name="", category_column="modCategory"):
    if game_name and "game" in df.columns:
        df = df[df["game"] == game_name]

    category_count = df[category_column].value_counts()
    download_sums = df.groupby(category_column)["downloads"].sum()
    
    category_per_download = (download_sums).sort_values(ascending=False)

    return category_per_download

def get_downloads_num_mods_category_ratio(df, game_name="", category_column="modCategory"):
    if game_name and "game" in df.columns:
        df = df[df["game"] == game_name]

    category_count = df[category_column].value_counts()
    download_sums = df.groupby(category_column)["downloads"].sum()

    category_ratio = (download_sums / category_count).sort_values(ascending=False)

    return category_ratio

def get_relative_demand(df, game_name="", category_column="modCategory"):
    if game_name and "game" in df.columns:
        df = df[df["game"] == game_name]

    mod_share = (
        df[category_column]
        .value_counts(normalize=True)
    )

    download_share = (
        df.groupby(category_column)["downloads"]
        .sum()
        / df["downloads"].sum()
    )

    relative_demand = (download_share / mod_share).sort_values(ascending=False)

    return relative_demand

def list_all_categories(df):
    categories = df["modCategory"].unique()
    return categories

def dump_category_names(df, category_column="modCategory", game_name=""):
    # per game 
    categories = df[category_column].unique()
    with open(f"data/categories.txt", "w") as f:
        # f.write(f"Categories for {game_name.capitalize()}:\n")
        for category in categories:
            f.write(f"{category}\n")
    print(f"Category names dumped to categories.txt")

def preprocess_mod_data(df):
    df = df.copy()

    # Convert dates
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], errors="coerce")

    # Fill numeric missing values
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
    df["endorsements"] = pd.to_numeric(df["endorsements"], errors="coerce").fillna(0)

    df["endorsement_ratio"] = np.where(
        df["downloads"] > 0,
        df["endorsements"] / df["downloads"],
        0
    )

    # modCategory: convert dict to category name
    df["modCategory"] = df["modCategory"].apply(
    lambda x: x["name"] if isinstance(x, dict) and "name" in x else "Unknown"
)

    df["description"] = df["description"].fillna("")
    df["summary"] = df["summary"].fillna("")

    # tags: convert list of dicts to comma-separated string of tag names
    df["tags"] = df["tags"].apply(
        lambda x: ", ".join([tag["name"] for tag in x]) if isinstance(x, list) else ""
    )

    return df

def create_category_mapping():
    with open("category_mapping.txt", "r") as f:
        mapping = [line.strip() for line in f.readlines()]

    mapping_dict = {}
    for mapping_entry in mapping:
        old_category, new_category = mapping_entry.split(">")
        old_category = old_category.strip()
        new_category = new_category.strip()
        mapping_dict[old_category] = new_category
    return mapping_dict

def map_category_names(df, category_column="modCategory"):
    # load from .txt, strip on >
    mapping_dict = create_category_mapping()
    # map old category names to new category names
    df['broadCategory'] = df['modCategory'].map(mapping_dict)
    return df

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # unescape html entities
    text = html.unescape(text)

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove html tags
    text = re.sub(r"<[^>]+>", " ", text)

    # remove bbcode-like tags: [url], [/url], [img], etc.
    text = re.sub(r"\[/?[^\]]+\]", " ", text)

    # remove leftover html-ish words that often survive scraping
    text = re.sub(r"\b(?:url|img|href|http|https|www|com)\b", " ", text)

    # remove punctuation / extra whitespace
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

def tokenize(text, stopwords=None):
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
    return tokens



def get_recurrent_terms(df, game_name="", category_column="modCategory", categories=None, 
    text_columns=("summary", "description", "tags"), ngram_range=(1, 2), top_n=20, min_freq=2, extra_stopwords=None):
    working_df = df.copy()

    if game_name and "game" in working_df.columns:
        working_df = working_df[working_df["game"] == game_name]

    if categories is not None:
        if isinstance(categories, str):
            categories = [categories]
        working_df = working_df[working_df[category_column].isin(categories)]

    # Combine text columns safely
    combined_text = (
        working_df.loc[:, list(text_columns)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )

    # Stopwords
    stopwords = set(ENGLISH_STOP_WORDS)
    stopwords.update({
        "mod", "mods", "game", "games", "adds", "add", "new", "also",
        "one", "two", "use", "using", "used", "make", "makes", "made",
        "change", "changes", "version", "updated", "update", "includes",
        "include", "support", "supports", "compatible", "replaces",
        "replace", "patch", "patches", "player", "players"
    })

    if extra_stopwords:
        stopwords.update(set(extra_stopwords))

    counter = Counter()

    min_n, max_n = ngram_range

    for text in combined_text:
        tokens = tokenize(text, stopwords=stopwords)
        for n in range(min_n, max_n + 1):
            if len(tokens) < n:
                continue
            ngrams = zip(*[tokens[i:] for i in range(n)])
            for gram in ngrams:
                term = " ".join(gram)
                counter[term] += 1

    # Apply minimum frequency threshold
    filtered = {term: count for term, count in counter.items() if count >= min_freq}

    return pd.Series(filtered).sort_values(ascending=False).head(top_n)