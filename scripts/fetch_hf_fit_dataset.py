"""Download cnamuangtoun/resume-job-description-fit into data/raw/.

Run once, locally, with the `datasets` package installed:
    pip install datasets
    python scripts/fetch_hf_fit_dataset.py
"""
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")

    train_path = OUT_DIR / "hf_fit_train.csv"
    test_path = OUT_DIR / "hf_fit_test.csv"
    ds["train"].to_csv(train_path)
    ds["test"].to_csv(test_path)

    print(f"train: {len(ds['train'])} rows -> {train_path}")
    print(f"test:  {len(ds['test'])} rows -> {test_path}")
    print(f"columns: {ds['train'].column_names}")


if __name__ == "__main__":
    main()