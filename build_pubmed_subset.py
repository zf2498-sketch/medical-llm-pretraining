import json


INPUT_FILE = "data/pubmed_all.jsonl"
OUTPUT_FILE = "data/pubmed_v1_50000.jsonl"
TARGET_COUNT = 50000


def main():
    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "").strip()

            if len(text) < 50:
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count >= TARGET_COUNT:
                break

    print("=" * 50)
    print(f"Saved subset with {count} records")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
