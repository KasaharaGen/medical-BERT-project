import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/{}/fullTextXML"
OUTPUT_PATH = "../data/epmc_fulltext.csv"
METADATA_PATH = "../data/metadata/epmc_filtered_metadata.csv"

KEYWORDS = [
    "dengue", "covid19", "full", "malaria", "sars coronavirus", "mars coronavirus"
]
KEYWORDS_NORMALIZED = [kw.lower() for kw in KEYWORDS]

if not os.path.isdir(os.path.dirname(OUTPUT_PATH)):
    raise FileNotFoundError("[ERROR] ../data ディレクトリが存在しません")

def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())

def contains_keywords(title, abstract):
    norm_text = normalize_text(title) + " " + normalize_text(abstract)
    return any(kw in norm_text for kw in KEYWORDS_NORMALIZED)

def fetch_fulltext_xml(source, pmcid):
    try:
        url = FULLTEXT_URL.format(source, pmcid)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "lxml-xml")
            body = soup.find("body") or soup.find("article")
            if body:
                paragraphs = body.find_all(["p", "sec", "title"])
                return "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        return ""
    except Exception as e:
        print(f"[ERROR] XML fetch failed: {e}")
        return ""

def fetch_html_text(pmcid):
    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MyResearchBot/1.0)",
            "Referer": "https://www.ncbi.nlm.nih.gov/"
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.select("#main-content p, div.tsec p")
            return "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        return ""
    except Exception as e:
        print(f"[ERROR] HTML fetch failed: {e}")
        return ""

def fetch_epmc_metadata(query, max_results=1000, page_size=1000):
    cursor = "*"
    results = []
    with tqdm(total=max_results, desc="Fetching metadata") as pbar:
        while len(results) < max_results:
            params = {
                "query": query,
                "format": "json",
                "pageSize": min(page_size, max_results - len(results)),
                "cursorMark": cursor,
                "resultType": "core"
            }
            try:
                r = requests.get(BASE_URL, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"[ERROR] Metadata fetch failed: {e}")
                break

            papers = data.get("resultList", {}).get("result", [])
            if not papers:
                break
            results.extend(papers)
            pbar.update(len(papers))

            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

    return results[:max_results]

def process_record(record):
    try:
        title = record.get("title", "")
        abstract = record.get("abstractText", "")
        source = record.get("source", "")
        pmcid = record.get("pmcid", "").replace("PMC", "")

        if not pmcid or not contains_keywords(title, abstract):
            return None

        full_text = fetch_fulltext_xml(source, pmcid)
        if not full_text:
            full_text = fetch_html_text(pmcid)
        if not full_text:
            print(f"[WARN] No fulltext → fallback for: {title[:60]}...")
            full_text = (title or "") + "\n\n" + (abstract or "")

        record["fulltext"] = full_text
        return record
    except Exception as e:
        print(f"[ERROR] Failed to process record: {e}")
        return None

if __name__ == "__main__":
    query = 'LICENSE:"CC-BY" AND OPEN_ACCESS:"y" AND PUB_YEAR:[2015 TO *]'
    metadata = fetch_epmc_metadata(query)
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(METADATA_PATH, index=False)
    print(f"📝 メタデータ保存完了: {METADATA_PATH}")

    all_records = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_record, record) for record in metadata]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing fulltext"):
            result = future.result()
            if result:
                all_records.append(result)

    pd.DataFrame(all_records).to_csv(OUTPUT_PATH, index=False)
    print(f"✅ フルテキスト保存完了: {OUTPUT_PATH}")

