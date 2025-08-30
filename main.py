# 2025年8月25日，上海市出台《关于优化调整本市房地产政策措施的通知》，想了解政策发布后的网络舆情。
# https://fgj.sh.gov.cn/gzdt/20250825/fe28d1dd27214dc6823dedfd9ccd5b97.html

import requests
from bs4 import BeautifulSoup
import chardet
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

# =====================
# 1. 数据采集模块
# =====================

def send_request(url, headers=None):
    """向URL发送请求并返回BeautifulSoup对象"""
    soup = None
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        result = chardet.detect(response.content)
        encoding = result['encoding']
        response.encoding = encoding
        soup = BeautifulSoup(response.text, 'html.parser')
    else:
        print(f"请求 {url} 失败，状态码: {response.status_code}")
    return soup

def extract_news_titles(soup, tag='a', pattern=r'.*'):
    """从HTML中提取符合pattern的标题"""
    titles = []
    if soup:
        for element in soup.find_all(tag):
            text = element.get_text(strip=True)
            if re.match(pattern, text):
                titles.append(text)
    return titles

def collect_data_from_platforms(keyword):
    """从多个平台收集新闻/帖子标题"""
    headers = {"User-Agent": "Mozilla/5.0"}
    data = {}

    # 平台1: 百度新闻搜索
    url_baidu = f"https://www.baidu.com/s?wd={keyword}"
    soup_baidu = send_request(url_baidu, headers=headers)
    data["baidu"] = extract_news_titles(soup_baidu, tag='a')

    # 平台2: 搜狐新闻搜索
    url_sohu = f"https://search.sohu.com/?keyword={keyword}"
    soup_sohu = send_request(url_sohu, headers=headers)
    data["sohu"] = extract_news_titles(soup_sohu, tag='a')

    # 平台3: 新浪新闻搜索
    url_sina = f"https://search.sina.com.cn/?q={keyword}"
    soup_sina = send_request(url_sina, headers=headers)
    data["sina"] = extract_news_titles(soup_sina, tag='a')

    # TODO: 微博、知乎、抖音等平台需要登录或API支持，这里仅保留结构接口
    data["weibo"] = [f"模拟微博帖子: {keyword}相关讨论"]
    data["zhihu"] = [f"模拟知乎问答: {keyword}的看法"]

    return data

# =====================
# 2. 情感分析模块
# =====================

classifier = pipeline("sentiment-analysis", model="IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment")

def analyze_sentiment(texts, platform_name="unknown"):
    """对文本列表进行情感分析，返回结果DataFrame"""
    results = []
    for text in texts:
        try:
            outputs = classifier(text[:512])
            label = outputs[0]['label']
            score = outputs[0]['score']
            results.append({"platform": platform_name, "text": text, "label": label, "score": score})
        except Exception as e:
            print(f"分析失败: {e}")
    return pd.DataFrame(results)

# =====================
# 3. 指数计算模块
# =====================

def compute_sentiment_index(df):
    if df.empty:
        return 0
    pos = len(df[df['label'].str.contains('positive')])
    neg = len(df[df['label'].str.contains('negative')])
    total = len(df)
    return (pos - neg) / total if total > 0 else 0

# =====================
# 4. 可视化模块
# =====================

def plot_sentiment_distribution(df):
    counts = df['label'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("情感分布")
    plt.show()

def generate_wordcloud(df):
    text_all = " ".join(df['text'].tolist())
    wc = WordCloud(font_path='simhei.ttf', background_color='white', width=800, height=400).generate(text_all)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("高频词词云")
    plt.show()

# =====================
# 5. 主流程示例
# =====================
if __name__ == "__main__":
    keyword = "上海楼市新政"
    platform_data = collect_data_from_platforms(keyword)

    all_dfs = []
    for platform, texts in platform_data.items():
        print(f"平台 {platform} 采集到 {len(texts)} 条")
        df_platform = analyze_sentiment(texts, platform_name=platform)
        all_dfs.append(df_platform)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(df_all.head())

    sentiment_index = compute_sentiment_index(df_all)
    print(f"总体情感指数: {sentiment_index:.2f}")

    if not df_all.empty:
        plot_sentiment_distribution(df_all)
        generate_wordcloud(df_all)
