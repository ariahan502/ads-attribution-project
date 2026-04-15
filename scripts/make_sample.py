from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/criteo_attribution_dataset.tsv.gz")
OUT_PATH = Path("data/samples/sample_1m.parquet")

def main():
    print("Loading raw data...")

    df = pd.read_csv(
        RAW_PATH,
        sep="\t",
        compression="gzip",
        nrows=1_000_000
    )

    print(f"Loaded shape: {df.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUT_PATH, index=False)

    print(f"Saved sample to: {OUT_PATH}")

if __name__ == "__main__":
    main()