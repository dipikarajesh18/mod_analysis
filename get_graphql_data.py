import os
import json
import time
import requests
from pathlib import Path
import dotenv

dotenv.load_dotenv()

api_key = os.environ.get("APIKEY")
URL = "https://api.nexusmods.com/v2/graphql"

HEADERS = {
    "apikey": api_key,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

QUERY = """
query GetModsForGame($gameDomain: String!, $offset: Int!, $count: Int!) {
  mods(
    filter: {
      gameDomainName: [{ value: $gameDomain }]
    }
    offset: $offset
    count: $count
  ) {
    totalCount
    nodes {
      modId
      name
      summary
      description
      downloads
      endorsements
      createdAt
      updatedAt

      tags {
        id
        name
      }

      modCategory {
        id
        name
      }

      uploader {
        memberId
        name
      }

      category
    }
  }
}
"""

def fetch_mod_page(game_domain, offset, count):
    variables = {
        "gameDomain": game_domain,
        "offset": offset,
        "count": count,
    }

    response = requests.post(
        URL,
        headers=HEADERS,
        json={"query": QUERY, "variables": variables},
        timeout=60,
    )

    response.raise_for_status()
    payload = response.json()    

    return payload["data"]["mods"]

def fetch_all_mods_for_game(game_domain: str, batch_size: int = 100, delay: float = 0.5):
    all_mods = []
    offset = 0
    total_count = None

    while True:
        page = fetch_mod_page(game_domain, offset, batch_size)
        mods = page["nodes"]

        if total_count is None:
            total_count = page["totalCount"]
            print(f"Total mods for {game_domain}: {total_count}")

        if not mods:
            break

        all_mods.extend(mods)
        print(f"Fetched {len(all_mods)} / {total_count}")

        offset += len(mods)

        if len(all_mods) >= total_count:
            break

        time.sleep(delay)

    return all_mods

def save_mods(game_domain: str, mods: list[dict]):
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{game_domain}_mods.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mods, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(mods)} mods to {out_path}")

if __name__ == "__main__":
    game_domain = "skyrim" #.change per game
    mods = fetch_all_mods_for_game(game_domain, batch_size=100)
    save_mods(game_domain, mods)