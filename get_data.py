import requests
import json
import os
import dotenv

dotenv.load_dotenv()

api_key = os.environ.get("APIKEY")
base_url = "https://api.nexusmods.com/v2/graphql"
headers = {
    "apikey": api_key,
    "Content-Type": "application/json",
}

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def get_games():
    url = f"{base_url}v1/games.json"
    response = requests.get(url, headers=headers)
    games = response.json()
    with open("data/games.json", "w") as f:
        json.dump(games, f, indent=2)


def get_game_details(game_name):
    url = f"{base_url}v1/games/{game_name}.json"
    response = requests.get(url, headers=headers)
    game_details = response.json()
    with open(f"data/{game_name}_details.json", "w") as f:
        json.dump(game_details, f, indent=2)


def get_mod(game_name, id):
    url = f"{base_url}v1/games/{game_name}/mods/{id}.json"
    response = requests.get(url, headers=headers)
    mod_details = response.json()
    return mod_details



def get_all_mods(game_name):
    game_json = read_json(f"data/{game_name}_details.json")
    num_mods = game_json.get("mods", 0)
    # print(num_mods)

    all_mods = []
    # for i in range(1, num_mods + 1):
    for i in range(1, 11):
        mod_details = get_mod(game_name, i)
        all_mods.append(mod_details)

    with open(f"data/mod_data/{game_name}_mod_details.json", "w") as f:
        json.dump(all_mods, f, indent=2)

# get_games()

# games = ["skyrim", "stardewvalley", "minecraft", "cyberpunk2077"]

games = ["stardewvalley"]

# for game in games:
#     get_all_mods(game)

# get_games()

# for game in games:
#     get_game_details(game)


query = """
query {
  mods(count: 10000) {
    nodes {
      modId
      name
      downloads
    }
  }
}
"""

response = requests.post(
    url,
    headers=headers,
    json={"query": query}
)

response.raise_for_status()

data = response.json()

print(json.dumps(data, indent=2))

