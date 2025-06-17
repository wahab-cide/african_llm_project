from datasets import load_dataset
from pathlib import Path

out_dir = Path("data/raw/twi/cc100")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("cc100", "ak", split="train", streaming=False)  
with open(out_dir / "ak.txt", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(row["text"] + "\n")

print("Twi CC-100 saved to", out_dir / "ak.txt")
