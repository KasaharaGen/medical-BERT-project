import requests
import time
import csv
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

BASE_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_KEYWORDS = [
    "dengue", "covid19", "malaria", "sars coronavirus", "mars coronavirus"
]

HEADERS = ["pmid", "title", "abstract", "pub_date", "journal"]


def build_query(keywords):
    return " OR ".join([f"{kw}[Title/Abstract]" for kw in keywords])

def fetch_pmids(query, retmax=10000):
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax
    }
    r = requests.get(BASE_ESEARCH, params=params)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]

def fetch_details(pmids):
    pmid_chunks = [pmids[i:i+200] for i in range(0, len(pmids), 200)]
    records = []
    for chunk in tqdm(pmid_chunks, desc="Fetching abstracts"):
        ids = ",".join(chunk)
        params = {
            "db": "pubmed",
            "id": ids,
            "retmode": "xml"
        }
        r = requests.get(BASE_EFETCH, params=params)
        r.raise_for_status()
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
                print(f"[WARN] Failed to parse one record: {e}")
        time.sleep(0.3)  # NCBI rate limit
    return records

def save_to_csv(records, path):
    with open(path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(records)
    print(f"✅ 保存完了: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default=",", help="カンマ区切りキーワード")
    parser.add_argument("--output", type=str, default="../data/pubmed_abstracts.csv")
    parser.add_argument("--retmax", type=int, default=5000)
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or DEFAULT_KEYWORDS
    query = build_query(keywords)

    print(f"🔍 検索クエリ: {query}")
    pmids = fetch_pmids(query, retmax=args.retmax)
    print(f"🔢 論文件数: {len(pmids)} 件")

    records = fetch_details(pmids)
    save_to_csv(records, args.output)

if __name__ == "__main__":
    main()
