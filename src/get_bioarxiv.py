import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.biorxiv.org/details/biorxiv/2015-01-01/2025-04-24/{}"
MAX_RESULTS = 10000000
PAGE_SIZE = 100
KEYWORDS = ["dengue", "covid19", "malaria", "full", "sars coronavirus", "mars coronavirus"]
KEYWORDS_NORMALIZED = [kw.lower() for kw in KEYWORDS]
OUTPUT_PATH = "../data/biorxiv_fulltext.csv"
METADATA_PATH = "../data/metadata/biorxiv_filtered_metadata.csv"

if not os.path.isdir(os.path.dirname(OUTPUT_PATH)):
    raise FileNotFoundError("[ERROR] ../data ディレクトリが存在しません")

def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())

def contains_keywords(title, abstract):
    norm_text = normalize_text(title) + " " + normalize_text(abstract)
    return any(kw in norm_text for kw in KEYWORDS_NORMALIZED)

def retry_request(url, headers=None, timeout=10, max_retries=3, sleep_time=5):
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(url, headers=headers, timeout=timeout)
            res.raise_for_status()
            return res
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"[WARN] Network issue at attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] All network retries failed for URL: {url}")
                return None
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Other request error: {e}")
            return None

def fetch_jats_xml(jats_url):
    res = retry_request(jats_url, timeout=10)
    return res.content if res else None

def fetch_html_text(html_url):
    res = retry_request(html_url, timeout=10)
    if not res:
        return ""
    soup = BeautifulSoup(res.text, "html.parser")
    paragraphs = soup.select("div.section p, div.abstract p")
    return "\n\n".join(p.get_text(strip=True) for p in paragraphs)

def extract_main_text_from_jats(xml_content, debug_title=""):
    try:
        soup = BeautifulSoup(xml_content, "lxml-xml")
        body = soup.find("body") or soup.find("article") or soup.find("sub-article")
        if not body:
            print(f"[WARN] <body>/<article> not found for: {debug_title[:60]}...")
            return ""
        paragraphs = body.find_all(["p", "sec", "title"])
        return "\n\n".join(tag.get_text(separator=" ", strip=True) for tag in paragraphs)
    except Exception as e:
        print(f"[ERROR] Failed to parse JATS XML for: {debug_title[:60]}... ({e})")
        return ""

def process_record(record):
    try:
        if record.get("license", "").lower() != "cc_by":
            return None
        if not contains_keywords(record.get("title", ""), record.get("abstract", "")):
            return None

        full_text = ""
        jats_url = record.get("jatsxml")
        if jats_url:
            xml = fetch_jats_xml(jats_url)
            if xml:
                full_text = extract_main_text_from_jats(xml, debug_title=record.get("title", ""))

        if not full_text:
            html_url = record.get("rel_link") or record.get("url")
            if html_url:
                full_text = fetch_html_text(html_url)

        if not full_text:
            print(f"[WARN] Fulltext is empty: {record.get('title', '')[:60]}... → fallback to title+abstract")
            full_text = (record.get("title") or "") + "\n\n" + (record.get("abstract") or "")

        record["fulltext"] = full_text
        return record
    except Exception as e:
        print(f"[ERROR] Failed to process record: {e}")
        return None

def main():
    all_records = []
    all_filtered_metadata = []
    for cursor in range(0, MAX_RESULTS, PAGE_SIZE):
        url = BASE_URL.format(cursor)
        print(f"[INFO] Fetching: {url}")
        response = retry_request(url, timeout=15)
        if response is None:
            break

        try:
            data = response.json()
        except Exception as e:
            print(f"[ERROR] JSON decode failed: {e}")
            break

        records = data.get("collection", [])
        if not records:
            print("[INFO] 空のレスポンス。終了します。")
            break

        filtered = [record for record in records if record.get("license", "").lower() == "cc_by" and contains_keywords(record.get("title", ""), record.get("abstract", ""))]
        all_filtered_metadata.extend(filtered)

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(process_record, record) for record in filtered]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_records.append(result)

        print(f"[INFO] {cursor + PAGE_SIZE}件目まで確認中（現在 {len(all_records)} 件収集済み）")
        time.sleep(0.5)

        if len(records) < PAGE_SIZE:
            print("[INFO] 最終ページに到達。処理を終了します。")
            break

    pd.DataFrame(all_records).to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(all_filtered_metadata).to_csv(METADATA_PATH, index=False)
    print(f"✅ 保存完了: {len(all_records)} 件のレコードを {OUTPUT_PATH} に出力")
    print(f"📝 メタデータ保存完了: {len(all_filtered_metadata)} 件を {METADATA_PATH} に出力")

if __name__ == "__main__":
    main()