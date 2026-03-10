import os
import pandas as pd
import requests
import json
from datasets import load_dataset
from config import DATA_DIR


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading WinoBias...")
    rows = []
    for subset in ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]:
        try:
            ds = load_dataset("uclanlp/wino_bias", subset, split="validation")
            for item in ds:
                rows.append({"sentence": " ".join(item["tokens"]), "subset": subset})
        except Exception as e:
            print(f"  Warning: {subset} failed - {e}")
    if rows:
        pd.DataFrame(rows).to_csv(f"{DATA_DIR}/winobias.csv", index=False)
        print(f"  Saved {len(rows)} rows")

    print("Downloading BBQ...")
    url = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Gender_identity.jsonl"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        rows = []
        for line in resp.text.strip().split("\n"):
            if line.strip():
                item = json.loads(line)
                rows.append({
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "ans0": item.get("ans0", ""),
                    "ans1": item.get("ans1", ""),
                    "ans2": item.get("ans2", ""),
                })
        pd.DataFrame(rows).to_csv(f"{DATA_DIR}/bbq.csv", index=False)
        print(f"  Saved {len(rows)} rows")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("Downloading StereoSet...")
    rows = []
    for config in ["intrasentence", "intersentence"]:
        try:
            ds = load_dataset("McGill-NLP/stereoset", config, split="validation")
            for item in ds:
                if item.get("bias_type") in ["gender", "profession"]:
                    sentences = item.get("sentences", {})
                    if isinstance(sentences, dict):
                        for sent, label in zip(
                                sentences.get("sentence", []),
                                sentences.get("gold_label", [])
                        ):
                            rows.append({
                                "context": item.get("context", ""),
                                "completion": sent,
                                "label": label,
                                "bias_type": item.get("bias_type", ""),
                            })
        except Exception as e:
            print(f"  Warning: {config} failed - {e}")
    if rows:
        pd.DataFrame(rows).to_csv(f"{DATA_DIR}/stereoset.csv", index=False)
        print(f"  Saved {len(rows)} rows")

    print("\nVerification:")
    all_good = True
    for filename in ["winobias.csv", "bbq.csv", "stereoset.csv"]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  {filename}: {len(df)} rows - OK")
        else:
            print(f"  {filename}: MISSING")
            all_good = False
    print("\nAll datasets ready!" if all_good else "\nSome datasets missing!")


if __name__ == "__main__":
    main()
