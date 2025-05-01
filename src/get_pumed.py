import requests
import time
import csv
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import json
import re

BASE_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_KEYWORDS = ["dengue", "covid19", "malaria", "sars coronavirus", "mars coronavirus"]
HEADERS = ["pmid", "title", "abstract", "pub_date", "journal"]

def build_query(keywords):
    return " OR ".join([f"{kw}[Title/Abstract]" for kw in keywords])

def clean_json_text(text):
    return re.sub(r"[\x00-\x1F\x7F]", "", text)

def safe_json_parse(response):
    try:
        raw_text = response.content.decode("utf-8", errors="ignore")
        cleaned = clean_json_text(raw_text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode failed: {e}\nRaw snippet: {response.text[:200]}...")
        raise

def request_with_retry(url, params, max_retries=3, sleep=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            if "html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("HTML returned instead of JSON: likely NCBI error page")
            return response
        except Exception as e:
            print(f"[WARN] Retry {attempt+1}/{max_retries} failed: {e}")
            time.sleep(sleep)
    raise RuntimeError(f"❌ 全ての再試行に失敗: {url}")

def init_search(query):
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 1,  # 履歴生成のため最低1件取得
        "usehistory": "y"
    }
    r = request_with_retry(BASE_ESEARCH, params)
    data = safe_json_parse(r)["esearchresult"]
    return int(data["count"]), data["webenv"], data["querykey"]

def fetch_pmids_from_history(webenv, query_key, retstart, retmax):
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json"
    }
    r = request_with_retry(BASE_ESEARCH, params)
    data = safe_json_parse(r)
    return data.get("esearchresult", {}).get("idlist", [])

def fetch_details(pmids):
    records = []
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i+200]
        ids = ",".join(chunk)
        params = {
            "db": "pubmed",
            "id": ids,
            "retmode": "xml"
        }
        try:
            r = request_with_retry(BASE_EFETCH, params)
            root = ET.fromstring(r.text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid = article.findtext(".//PMID")
                    title = article.findtext(".//ArticleTitle")
                    abstract = " ".join([abst.text or "" for abst in article.findall(".//AbstractText")])
                    pub_date = article.findtext(".//PubDate/Year") or ""
                    journal = article.findtext(".//Journal/Title") or ""
                    records.append([pmid, title, abstract, pub_date, journal])
                except Exception as e:
                    print(f"[WARN] Failed to parse record: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch efetch even after retries: {e}")
        time.sleep(0.3)
    return records

def save_to_csv(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(records)
    print(f"✅ 保存完了: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default=",", help="カンマ区切りキーワード")
    parser.add_argument("--output", type=str, default="../data/pubmed_abstracts.csv")
    parser.add_argument("--retmax", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or DEFAULT_KEYWORDS
    query = build_query(keywords)

    print(f"🔍 検索クエリ: {query}")
    count, webenv, query_key = init_search(query)
    count = min(count, args.retmax)
    print(f"🔢 総件数: {count} 件（上限: {args.retmax}）")

    all_records = []
    for retstart in tqdm(range(0, count, args.batch), desc="Downloading PMIDs"):
        print(f"📥 Fetching retstart={retstart} ...")
        pmids = fetch_pmids_from_history(webenv, query_key, retstart, args.batch)
        print(f"🔢 Got {len(pmids)} PMIDs")
        if not pmids:
            break
        records = fetch_details(pmids)
        all_records.extend(records)

    save_to_csv(all_records, args.output)

if __name__ == "__main__":
    main()