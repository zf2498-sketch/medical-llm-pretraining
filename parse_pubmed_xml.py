import os
import gzip
import json
import xml.etree.ElementTree as ET


INPUT_DIR = "data/pubmed_baseline"
OUTPUT_FILE = "data/pubmed_all.jsonl"


def get_text(element):
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def extract_article_text(pubmed_article):
    medline = pubmed_article.find("MedlineCitation")
    if medline is None:
        return None

    pmid_elem = medline.find("PMID")
    pmid = get_text(pmid_elem)

    article = medline.find("Article")
    if article is None:
        return None

    title = get_text(article.find("ArticleTitle"))

    abstract_elem = article.find("Abstract")
    if abstract_elem is None:
        return None

    abstract_parts = []
    for abstract_text in abstract_elem.findall("AbstractText"):
        txt = get_text(abstract_text)
        if txt:
            label = abstract_text.attrib.get("Label", "").strip()
            if label:
                txt = f"{label}: {txt}"
            abstract_parts.append(txt)

    abstract = " ".join(abstract_parts).strip()

    if not abstract:
        return None

    full_text = f"{title}. {abstract}".strip() if title else abstract
    if len(full_text) < 50:
        return None

    return {
        "pmid": pmid,
        "text": full_text
    }


def parse_one_file(filepath, writer):
    count = 0
    with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
        tree = ET.parse(f)
        root = tree.getroot()

        for article in root.findall("PubmedArticle"):
            record = extract_article_text(article)
            if record is not None:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


def main():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    files = sorted(
        [
            os.path.join(INPUT_DIR, name)
            for name in os.listdir(INPUT_DIR)
            if name.endswith(".xml.gz")
        ]
    )

    if not files:
        raise FileNotFoundError(f"No .xml.gz files found in {INPUT_DIR}")

    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as writer:
        for fp in files:
            print(f"Parsing: {fp}")
            cnt = parse_one_file(fp, writer)
            total += cnt
            print(f"  extracted {cnt} records")

    print("=" * 50)
    print(f"Done. Total extracted records: {total}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
