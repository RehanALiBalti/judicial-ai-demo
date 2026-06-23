import re
import requests
from bs4 import BeautifulSoup

r = requests.get(
    "https://fccp.gov.pk/judgments?page=1",
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    timeout=30,
)
soup = BeautifulSoup(r.text, "html.parser")
table = soup.find("table")
rows = table.find_all("tr") if table else []
print("rows", len(rows))
for tr in rows[1:4]:
    tds = tr.find_all("td")
    if not tds:
        continue
    texts = [td.get_text(" ", strip=True) for td in tds]
    a = tr.find("a", href=True)
    print(texts, a["href"] if a else None)

m = re.search(r"Showing.*?of\s*(\d+)\s*results", r.text, re.I | re.S)
print("total", m.group(1) if m else "na")

page_links = [a.get("href") for a in soup.find_all("a", href=True) if "judgments?page=" in a.get("href", "")]
print("page_links sample", sorted(set(page_links))[:8])
