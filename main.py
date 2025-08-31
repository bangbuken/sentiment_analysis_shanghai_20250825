import time
import random
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
from urllib.parse import quote
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import jieba

# ---------------------------
# 配置
# ---------------------------
FILTER_KEYWORDS = ["楼", "房", "政", "市", "购", "买", "贷", "价", "限", "利率", "公积金"]
EXCLUDE_KEYWORDS = ["游戏", "体育", "娱乐"]
TIME_LIMIT = datetime(2025, 8, 25)

# ---------------------------
# 1) 情感分析模型
# ---------------------------
# classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone-chinese")

def sentiment_score(text):
    outputs = classifier(text[:512])
    return outputs[0]['label'], outputs[0]['score']

# ---------------------------
# 2) Selenium 设置
# ---------------------------
def get_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# ---------------------------
# 3) 抓取搜索结果
# ---------------------------
def search_sina_news_selenium(keyword, max_pages=50):
    driver = get_driver()
    keyword_encoded = quote(keyword)
    base_url = f"https://search.sina.com.cn/?q={keyword_encoded}&c=news&from=channel"
    driver.get(base_url)
    time.sleep(random.uniform(0.2, 0.4))

    all_news = []
    seen_urls = set()

    for page in range(max_pages):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.box-result"))
            )
            items = driver.find_elements(By.CSS_SELECTOR, "div.box-result a[target='_blank']")
            for item in items:
                title = item.text.strip()
                url = item.get_attribute("href").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # 必须包含关键词“上海”
                if "上海" not in title:
                    continue

                if FILTER_KEYWORDS and not any(k in title for k in FILTER_KEYWORDS):
                    continue
                if EXCLUDE_KEYWORDS and any(k in title for k in EXCLUDE_KEYWORDS):
                    continue

                # 发布时间限制
                try:
                    pub_elem = item.find_element(By.XPATH, "./following-sibling::span")
                    pub_str = pub_elem.text.strip()
                    pub_date = datetime.strptime(pub_str.split()[0], "%Y-%m-%d")
                    if pub_date < TIME_LIMIT:
                        continue
                except:
                    pass

                all_news.append({"title": title, "url": url})
            print(f"第 {page+1} 页抓取到 {len(items)} 条新闻，累计 {len(all_news)} 条")
        except Exception as e:
            print(f"第 {page+1} 页异常: {e}")

        # 翻页
        try:
            next_btn = driver.find_element(By.LINK_TEXT, "下一页")
            if "javascript:void(0)" in next_btn.get_attribute("href"):
                print("已到最后一页")
                break
            next_btn.click()
            time.sleep(random.uniform(0.1, 0.5))
        except Exception:
            print("没有找到下一页按钮或已到最后一页")
            break

    driver.quit()
    return all_news

# ---------------------------
# 4) 获取新闻正文
# ---------------------------
def fetch_news_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return ""
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, "html.parser")
        candidates = [
            soup.find(id="artibody"),
            soup.find("div", class_="article"),
            soup.find("div", id="article"),
            soup.find("div", class_="newstext"),
            soup.find("article"),
        ]
        for tag in candidates:
            if tag:
                txt = tag.get_text(separator="\n", strip=True)
                if txt and len(txt) > 30:
                    return txt
        return soup.get_text(separator="\n", strip=True)
    except:
        return ""

# ---------------------------
# 5) 保存 Excel 并生成图表
# ---------------------------
def plot_sentiment_pie(data, filename="sentiment_pie.png"):
    labels = [row[3] for row in data]
    counter = Counter(labels)
    total = sum(counter.values())
    sizes = [v / total for v in counter.values()]
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=counter.keys(), autopct='%1.1f%%', startangle=90, colors=["#66c2a5","#fc8d62","#8da0cb"])
    plt.title("新闻情感比例")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"情感比例饼状图已保存到 {filename}")

def plot_title_wordcloud(data, filename="title_wordcloud.png"):
    titles = [row[1] for row in data]
    text = " ".join(titles)
    words = jieba.cut(text)
    text_cut = " ".join(words)
    stopwords = set(STOPWORDS)
    stopwords.update(["新闻","报道","表示","公司","上海","楼市","房地产"])
    wc = WordCloud(font_path="simhei.ttf", width=800, height=600, background_color="white", stopwords=stopwords)
    wc.generate(text_cut)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"标题词云已保存到 {filename}")

def save_to_excel(data, filename="sina_news.xlsx"):
    df = pd.DataFrame(data, columns=["新闻URL", "标题", "正文", "情感", "置信度"])
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"数据已保存到 {filename}")
    plot_sentiment_pie(data)
    plot_title_wordcloud(data)

# ---------------------------
# 6) 主流程
# ---------------------------
def run_pipeline(keywords=["上海房地产", "上海楼市新政", "限购政策"], max_pages_per_keyword=None):
    all_results = []
    seen_urls = set()

    for kw in keywords:
        print(f"\n=== 抓取关键词: {kw} ===")
        news_list = search_sina_news_selenium(kw, max_pages=max_pages_per_keyword)
        print(f"关键词[{kw}] 共抓到 {len(news_list)} 条候选链接")

        for news in news_list:
            if news['url'] in seen_urls:
                continue
            seen_urls.add(news['url'])
            content = fetch_news_content(news['url'])
            if not content:
                continue
            label, score = sentiment_score(news['title'] + " " + content)
            all_results.append([news['url'], news['title'], content, label, score])
            time.sleep(random.uniform(0.1, 0.5))

    save_to_excel(all_results)

# ---------------------------
# 7) 运行示例
# ---------------------------
if __name__ == "__main__":
    run_pipeline(keywords=[
        "上海房地产", "上海楼市", "上海房地产新政", "上海楼市新政", "上海限购政策",
        "沪六条", "上海825", "上海房地产调控", "上海外环", "上海取消限购",
        "上海楼市限购", "上海房贷", "上海买房", "上海公积金"
    ], max_pages_per_keyword=20)