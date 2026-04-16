import pandas as pd
import numpy as np
from pathlib import Path

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


def load_mod_data(game_name):
    path = Path(f"data/mods/{game_name}_mods.json")
    df = pd.read_json(path)
    return df


def preprocess_mod_data(df):
    df = df.copy()

    # Convert dates
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], errors="coerce")

    # Fill numeric missing values
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
    df["endorsements"] = pd.to_numeric(df["endorsements"], errors="coerce").fillna(0)

    # Faster than apply(...)
    df["endorsement_ratio"] = np.where(
        df["downloads"] > 0,
        df["endorsements"] / df["downloads"],
        0
    )

    df["modCategory"] = df["modCategory"].apply(
    lambda x: x["name"] if isinstance(x, dict) and "name" in x else "Unknown"
)
    # Useful derived features for later analysis
    now = pd.Timestamp.now()

    df["log_downloads"] = np.log1p(df["downloads"])
    df["log_endorsements"] = np.log1p(df["endorsements"])

    # df["days_since_created"] = (now - df["createdAt"]).dt.days
    # df["days_active"] = (df["updatedAt"] - df["createdAt"]).dt.days

    df["description"] = df["description"].fillna("")
    df["summary"] = df["summary"].fillna("")

    df["description_length"] = df["description"].str.len()
    df["summary_length"] = df["summary"].str.len()

    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    df["num_tags"] = df["tags"].apply(len)

    return df

import re

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove BBCode
    text = re.sub(r"\[/?[^\]]+\]", " ", text)

    # Remove HTML entities / leftovers
    text = re.sub(r"&\w+;?", " ", text)

    # Remove obvious formatting leftovers
    text = re.sub(r"\b(gt|lt|amp|quot|br|img|url|png|jpg|jpeg)\b", " ", text)

    # Remove file paths / install path words
    text = re.sub(r"\b(steamapps|common|mods|mod|folder|config|json|unzip|install|installation|download)\b", " ", text)

    # Remove framework boilerplate
    text = re.sub(r"\b(smapi|content|patcher|generic|menu|fashion|sense)\b", " ", text)

    # Remove punctuation / numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def analyze_mod_data(df):
    # Top 10 most downloaded mods
    top_downloaded = df.sort_values(by="downloads", ascending=False).head(10)
    print("\nTop 10 Most Downloaded Mods:")
    print(top_downloaded[["name", "downloads", "endorsements", "endorsement_ratio"]])

    # Correlation between downloads and endorsements
    correlation = df["downloads"].corr(df["endorsements"])
    print(f"\nCorrelation between downloads and endorsements: {correlation:.2f}")

    # Basic summary stats
    print("\nBasic summary statistics:")
    print(df[[
        "downloads",
        "endorsements",
        "endorsement_ratio",
        "log_downloads",
        "num_tags",
        "description_length",
        "days_active"
    ]].describe())


game_name = "stardewvalley"
df = load_mod_data(game_name)
df = preprocess_mod_data(df)

# analyze_mod_data(df)

# print("\nColumns:")
# print(df.columns.tolist())

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

DOMAIN_STOPWORDS = {
    "stardew", "valley", "mod", "mods", "game", "using", "use",
    "download", "install", "folder", "json", "config", "smapi",
    "content", "patcher", "version", "latest", "original",
    "added", "adds", "add", "make", "sure", "feel", "free",
    "let", "know", "run", "file", "files", "steamapps", "common",
    "generic", "menu", "source", "code", "special", "thanks",
    "hope", "enjoy", "true", "false", "nexus", "minecraft"
}

custom_stopwords = ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS)

def get_top_categories(df, n=20):
    category_counts = df["modCategory"].value_counts().head(n)
    return category_counts

# print("\nTop categories:")
# print(get_top_categories(df, 20))

# text = df["summary"].fillna("") + " " + df["description"].fillna("")
df["clean_text"] = (
    df["summary"].fillna("") + " " + df["description"].fillna("")
).apply(clean_text)

df["clean_len"] = df["clean_text"].str.split().str.len()

analysis_df = df[df["clean_len"] >= 20].copy()

TRANSLATION_WORDS = [
    "translation", "translate", "tradu", "traduccion", "tradução",
    "espanol", "español", "portugues", "português", "pt br",
    "francais", "deutsch", "italiano", "russian", "korean", "japanese"
]

pattern = "|".join(TRANSLATION_WORDS)

analysis_df = analysis_df[
    ~analysis_df["clean_text"].str.contains(pattern, case=False, na=False)
].copy()

vectorizer = CountVectorizer(
    stop_words=list(custom_stopwords),
    ngram_range=(2, 2),   # bigrams only
    min_df=20             # appears in at least 20 mods
)

X = vectorizer.fit_transform(analysis_df["clean_text"])

counts = X.sum(axis=0).A1
ngrams = vectorizer.get_feature_names_out()

ngram_counts = sorted(
    zip(ngrams, counts),
    key=lambda x: x[1],
    reverse=True
)

# print(ngram_counts[:50])

def top_ngrams_for_category(df, category, n=20):
    subset = df[df["modCategory"] == category].copy()

    if len(subset) < 20:
        return []

    vectorizer = CountVectorizer(
        stop_words=list(custom_stopwords),
        ngram_range=(2, 2),
        min_df=5,
        max_df=0.7
    )

    X = vectorizer.fit_transform(subset["clean_text"])
    counts = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()

    results = sorted(zip(ngrams, counts), key=lambda x: x[1], reverse=True)
    return results[:n]


JUNK_TERMS = {
    "di logos", "di logo", "kh ng", "ng th", "ng ng", "ch nh", "ch ng", "nh ng",
    "npc npc", "debug ebi", "drag drop", "technical requires", "list available",
    "don want", "enable disable", "change settings", "linux mac",
    "compatibility works", "compatibility patch", "single player", "player multiplayer"
}

def filter_ngrams(ngram_counts, junk_terms=JUNK_TERMS):
    return [(ngram, count) for ngram, count in ngram_counts if ngram not in junk_terms]

# print(filter_ngrams(top_ngrams_for_category(analysis_df, "Dialogue", 20)))
# print(filter_ngrams(top_ngrams_for_category(analysis_df, "Events", 20)))
# print(filter_ngrams(top_ngrams_for_category(analysis_df, "Gameplay Mechanics", 20)))


import re
import pandas as pd
import numpy as np

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[/?[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_mod_data(df):
    df = df.copy()

    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], errors="coerce")

    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
    df["endorsements"] = pd.to_numeric(df["endorsements"], errors="coerce").fillna(0)

    df["mod_category"] = df["modCategory"].apply(
        lambda x: x["name"] if isinstance(x, dict) and "name" in x else "Unknown"
    )

    df["endorsement_ratio"] = np.where(
        df["downloads"] > 0,
        df["endorsements"] / df["downloads"],
        0
    )

    df["summary"] = df["summary"].fillna("")
    df["description"] = df["description"].fillna("")

    df["search_text"] = (
        df["name"].fillna("") + " " +
        df["mod_category"].fillna("") + " " +
        df["summary"] + " " +
        df["description"]
    ).apply(clean_text)

    return df

# df = load_mod_data(game_name)
df = preprocess_mod_data(df)

QUEER_KEYWORDS = [
    "queer", "lgbt", "lgbtq", "lesbian", "gay", "bisexual", "trans",
    "transgender", "nonbinary", "non binary", "pronoun", "pronouns",
    "same sex", "same gender", "wlw", "mlm", "sapphic", "achillean"
]

ACCESSIBILITY_KEYWORDS = [
    "accessibility", "accessible", "colorblind", "color blind",
    "dyslexia", "screen reader", "high contrast", "subtitle", "subtitles",
    "larger text", "bigger font", "font size", "visual aid",
    "hearing", "deaf", "blind", "low vision", "photosensitive",
    "easy mode", "difficulty", "motor", "remap", "keybind", "keybinding"
]

def make_keyword_pattern(keywords):
    escaped = [re.escape(k) for k in keywords]
    return r"\b(" + "|".join(escaped) + r")\b"

queer_pattern = make_keyword_pattern(QUEER_KEYWORDS)
accessibility_pattern = make_keyword_pattern(ACCESSIBILITY_KEYWORDS)

df["is_queer"] = df["search_text"].str.contains(queer_pattern, regex=True, na=False)
df["is_accessibility"] = df["search_text"].str.contains(accessibility_pattern, regex=True, na=False)

print("Queer mods:", df["is_queer"].sum())
print("Accessibility mods:", df["is_accessibility"].sum())

print(df[df["is_queer"]][["name", "modCategory", "downloads", "endorsements"]].head(20))
print(df[df["is_accessibility"]][["name", "modCategory", "downloads", "endorsements"]].head(20))

def top_categories_for_subset(df, flag_col, n=15):
    return df[df[flag_col]]["modCategory"].value_counts().head(n)

print(top_categories_for_subset(df, "is_queer"))
print(top_categories_for_subset(df, "is_accessibility"))

def subset_stats(df, flag_col):
    subset = df[df[flag_col]]
    return {
        "count": len(subset),
        "avg_downloads": subset["downloads"].mean(),
        "median_downloads": subset["downloads"].median(),
        "avg_endorsements": subset["endorsements"].mean(),
        "avg_endorsement_ratio": subset["endorsement_ratio"].mean()
    }

print("Queer stats:", subset_stats(df, "is_queer"))
print("Accessibility stats:", subset_stats(df, "is_accessibility"))