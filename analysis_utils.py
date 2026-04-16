import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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

def plot_category_distribution(df, category_values, top_n=10, game_name="All 4 Games"):
    category_values.head(top_n).plot(kind="bar")
    plt.title(f"Top {top_n} Categories Distribution for {game_name}")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def get_top_categories(df, game_name="", category_column="modCategory", type="count"):
    if game_name:
        df = df[df["game"] == game_name]
    category_percent = (
        df[category_column]
        .value_counts(normalize=True)
        .mul(100)
    )  

    category_count = df[category_column].value_counts()

    # return category_count
    if type == "percent":
        return category_percent
    else:
        return category_count

def get_top_by_downloads(df, game_name="", category_column="modCategory"):
    if game_name:
        df = df[df["game"] == game_name]

    category_count = df[category_column].value_counts()
    download_sums = df.groupby(category_column)["downloads"].sum()
    
    category_per_download = (download_sums / category_count).sort_values(ascending=False)

    return category_per_download

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


games = ["stardewvalley", "minecraft", "skyrim", "cyberpunk2077"]
# games = ["stardewvalley"]

    game_df_list = {}
    category_counts = {}
    for game_name in games:
        df = load_mod_data_per_game(game_name)
        df = preprocess_mod_data(df)
        game_df_list[game_name] = df


all_df = combine_all_games_data(game_df_list)
dump_category_names(all_df, category_column="modCategory")
# print(get_top_categories(all_df, game_name="stardewvalley", category_column="modCategory"))
# print(get_top_by_downloads(all_df, game_name="stardewvalley", category_column="modCategory"))

    # all_df = combine_all_games_data(game_df_list)
    # print(get_top_categories(all_df, category_column="modCategory"))

    # print(all_df.head())

    #     # get_top_categories(df)
    #     # print(df.columns)
    #     # print("Number of rows for {}: {}".format(game_name, len(df)))
    #     # print(f"\nCategories for {game_name.capitalize()}:")
    #     # print(list_all_categories(df))




