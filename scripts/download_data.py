import urllib.request
from pathlib import Path 

raw_dir = Path("data/raw")
raw_dir.mkdir(parents = True, exist_ok = True)

seasons  = {

    "2020" : "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "2021" : "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022" : "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023" : "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024" : "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
}

for season, url in seasons.items():
    out_file = raw_dir / f"E0_{season}.csv"
    print(f"Downloading {season} -> {out_file}")
    urllib.request.urlretrieve(url, out_file)