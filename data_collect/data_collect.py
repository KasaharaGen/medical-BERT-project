import requests
import pandas as pd
import time
from Bio import Entrez

# -------------------------
# 設定
# -------------------------
EMAIL = "g-kasahara-9b9@eagle.sophia.ac.jp"  # ← ここを正しいメールに置き換え
USER_AGENT = f"MedBERT-Pretrainer/1.0 (mailto:{EMAIL})"
HEADERS = {"User-Agent": USER_AGENT}
Entrez.email = EMAIL
Entrez.tool = "MedBERT-Pretrainer"
API_SLEEP_TIME = 0.5
PAGE_SIZE = 1000
RETMAX_PUBMED = 500
MAX_PUBMED_RECORDS = 2000

# -------------------------
# Europe PMC クエリ
# -------------------------
QUERY_EPMC = """
("infectious disease*" OR "virus*" OR "bacteri*" OR "parasit*" OR "fung*")
AND PUB_YEAR:[2015 TO 2025]
"""

# -------------------------
# PubMed クエリ
# -------------------------
QUERY_PUBMED = """
(
  "infectious disease"[Title/Abstract] OR
  "communicable disease"[Title/Abstract] OR
  "virus"[Title/Abstract] OR
  "bacteria"[Title/Abstract] OR
  "parasite"[Title/Abstract] OR
  "fungal infection"[Title/Abstract]
)
AND ("2015"[PDAT] : "2025"[PDAT])
"""

# -------------------------
# Europe PMC 抽出
# -------------------------
def fetch_europe_pmc(query):
    print("[Europe PMC] 取得開始")
    results = []
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    cursor = "*"
    while True:
        params = {
            "query": query,
            "format": "json",
            "pageSize": PAGE_SIZE,
            "cursorMark": cursor,
            "resultType": "lite"
        }
        response = requests.get(base_url, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        page_results = data.get("resultList", {}).get("result", [])
        for entry in page_results:
            doi = entry.get("doi", "").lower()
            title = entry.get("title", "")
            abstract = entry.get("abstractText", "")
            fulltext = abstract
            results.append({
                "DOI": doi,
                "Title": title,
                "Abstract": abstract,
                "FullText": fulltext,
                "Source": "EuropePMC"
            })
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(API_SLEEP_TIME)
    print(f"[Europe PMC] 件数: {len(results)}")
    return results

# -------------------------
# PubMed 抽出（ページ対応）
# -------------------------
def fetch_pubmed_all(query, retmax=500, max_records=None):
    print("[PubMed] 全件数確認中...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    record = Entrez.read(handle)
    total_count = int(record["Count"])
    print(f"[PubMed] ヒット数: {total_count}")
    if max_records:
        total_count = min(total_count, max_records)

    all_results = []
    for start in range(0, total_count, retmax):
        print(f"[PubMed] {start+1} 件目から取得中...")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, retstart=start)
        record = Entrez.read(handle)
        id_list = record["IdList"]
        time.sleep(API_SLEEP_TIME)
        if not id_list:
            break
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        records = Entrez.read(handle)
        time.sleep(API_SLEEP_TIME)
        for article in records.get('PubmedArticle', []):
            citation = article['MedlineCitation']
            article_data = citation.get('Article', {})
            title = article_data.get('ArticleTitle', '')
            abstract = article_data.get('Abstract', {}).get('AbstractText', [""])
            abstract_text = " ".join(abstract) if isinstance(abstract, list) else abstract
            elocation = article_data.get('ELocationID', [])
            doi = ""
            if isinstance(elocation, list):
                for loc in elocation:
                    if loc.attributes.get('EIdType') == 'doi':
                        doi = str(loc).lower()
            elif hasattr(elocation, 'attributes') and elocation.attributes.get('EIdType') == 'doi':
                doi = str(elocation).lower()
            all_results.append({
                "DOI": doi,
                "Title": title,
                "Abstract": abstract_text,
                "FullText": abstract_text,
                "Source": "PubMed"
            })
    print(f"[PubMed] 件数: {len(all_results)}")
    return all_results

# -------------------------
# 実行
# -------------------------
def main():
    epmc_data = fetch_europe_pmc(QUERY_EPMC)
    pubmed_data = fetch_pubmed_all(QUERY_PUBMED, retmax=RETMAX_PUBMED, max_records=MAX_PUBMED_RECORDS)

    df_epmc = pd.DataFrame(epmc_data)
    df_pubmed = pd.DataFrame(pubmed_data)

    # 統合（DOIで重複削除）
    merged = pd.concat([df_epmc, df_pubmed], ignore_index=True)
    merged = merged.drop_duplicates(subset="DOI", keep="first")
    merged = merged[merged["DOI"].notnull() & (merged["DOI"] != "")]

    # 保存
    df_epmc.to_csv("infectious_disease_epmc.csv", index=False)
    df_pubmed.to_csv("infectious_disease_pubmed.csv", index=False)
    merged.to_csv("infectious_disease_merged.csv", index=False)

    print("\n✅ 保存完了:")
    print(" - infectious_disease_epmc.csv")
    print(" - infectious_disease_pubmed.csv")
    print(" - infectious_disease_merged.csv")
    print(f"📊 DOI重複排除後の論文数: {len(merged)}")

if __name__ == "__main__":
    main()
