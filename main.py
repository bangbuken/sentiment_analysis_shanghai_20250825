import time
import random
import pandas as pd
from datetime import datetime
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
from urllib.parse import quote
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import jieba
import re

# ---------------------------
# 配置
# ---------------------------
FILTER_KEYWORDS = ["楼", "房", "政", "市", "购", "买", "贷", "价", "限", "利率", "公积金"]
EXCLUDE_KEYWORDS = ["游戏", "体育", "娱乐"]
TIME_LIMIT = datetime(2025, 8, 25)

# 情感分析模型
classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone-chinese")

def sentiment_score(text):
    outputs = classifier(text[:512])
    return outputs[0]['label'], outputs[0]['score']

# ---------------------------
# 微博 Cookie 配置
# ---------------------------
WEIBO_COOKIES = {
    "SUB": "_2A25FsivMDeRhGedG41sQ9SnIyj-IHXVmziEErDV8PUNbmtANLUaskW9NUOk2bX3oBwt9lpeoTCjkFIufzwVlYmyA",
    "SUBP": "0033WrSXqPxfM725Ws9jqgMF55529P9D9WFj7cf7HiukpvQikGSgy-5z5NHD95Qp1hn4eK-NSh20Ws4DqcjMi--NiK.Xi-2Ri--ciKnRi-zN1KLDMN90UNeN15tt",
    # "SSOLoginState": "替换成你的SSOLoginState" # 不是必需
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Referer": "https://weibo.com/",
}

# ---------------------------
# Selenium 设置
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

    # 先打开一次微博主页，再注入 cookie
    driver.get("https://weibo.com/")
    time.sleep(3)
    for k, v in WEIBO_COOKIES.items():
        driver.add_cookie({"name": k, "value": v, "domain": ".weibo.com"})
    driver.refresh()
    time.sleep(2)
    return driver

# ---------------------------
# 1) 搜索微博帖子
# ---------------------------
def search_weibo_posts(keyword, max_pages=5):
    driver = get_driver()
    keyword_encoded = quote(keyword)
    base_url = f"https://s.weibo.com/weibo?q={keyword_encoded}"
    driver.get(base_url)
    time.sleep(random.uniform(0.5, 1.0))

    all_posts = []
    seen_ids = set()

    for page in range(max_pages):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.card-wrap"))
            )
            items = driver.find_elements(By.CSS_SELECTOR, "div.card-wrap")

            for item in items:
                try:
                    mid = item.get_attribute("mid")
                    if not mid or mid in seen_ids:
                        continue
                    seen_ids.add(mid)

                    # 帖子正文
                    content_elem = item.find_element(By.CSS_SELECTOR, "p.txt")
                    content = content_elem.text.strip()
                    if not content:
                        continue

                    if "上海" not in content:
                        continue
                    if FILTER_KEYWORDS and not any(k in content for k in FILTER_KEYWORDS):
                        continue
                    if EXCLUDE_KEYWORDS and any(k in content for k in EXCLUDE_KEYWORDS):
                        continue

                    # 发布时间
                    try:
                        time_elem = item.find_element(By.CSS_SELECTOR, "p.from a")
                        pub_str = time_elem.text.strip()
                        if "2025" in pub_str:
                            pub_date = datetime.strptime(pub_str.split()[0], "%Y-%m-%d")
                            if pub_date < TIME_LIMIT:
                                continue
                        else:
                            pub_date = None
                    except:
                        pub_date = None

                    all_posts.append({
                        "mid": mid,
                        "content": content,
                        "pub_date": pub_date
                    })
                except:
                    continue

            print(f"第 {page+1} 页抓取到 {len(all_posts)} 条帖子")

            # 翻页
            try:
                next_btn = driver.find_element(By.LINK_TEXT, "下一页")
                next_btn.click()
                time.sleep(random.uniform(1, 2))
            except:
                print("已到最后一页")
                break

        except Exception as e:
            print(f"第 {page+1} 页异常: {e}")
            break

    driver.quit()
    return all_posts

# ---------------------------
# 2) 抓取评论
# ---------------------------
def fetch_comments(mid, max_pages=3, count=20):
    """通过微博 JSON 接口抓取评论"""
    comments = []
    session = requests.Session()
    session.cookies.update(WEIBO_COOKIES)
    session.headers.update(HEADERS)

    for page in range(1, max_pages+1):
        url = f"https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id={mid}&is_show_bulletin=2&count={count}&page={page}"
        try:
            res = session.get(url, timeout=10)
            if res.status_code != 200:
                print(f"评论请求失败 mid={mid}, code={res.status_code}")
                break
            data = res.json()
            if "data" not in data or not data["data"]:
                break
            for c in data["data"]:
                text = c.get("text_raw", "").strip()
                if text:
                    comments.append(text)
        except Exception as e:
            print(f"抓取评论异常 mid={mid}, page={page}, err={e}")
            break
        time.sleep(random.uniform(0.5, 1.0))
    return comments

# ---------------------------
# 3) 保存 Excel 并生成图表
# ---------------------------
def plot_sentiment_pie(data, filename="weibo_sentiment_pie.png"):
    """
    统计 data 中 第4列(索引3) 的情感标签占比并画饼图
    data 的每行结构: [mid, 类型, 内容, 情感, 置信度]
    """
    labels = [row[3] for row in data if len(row) >= 4 and row[3]]
    counter = Counter(labels)
    total = sum(counter.values())
    if total == 0:
        print("没有可用于饼图的情感数据，跳过绘制。")
        return

    # 尽量按常见顺序展示；若不存在则按 counter 的键顺序
    pref_order = ["positive", "neutral", "negative", "POSITIVE", "NEUTRAL", "NEGATIVE", "正向", "中性", "负向"]
    keys = [k for k in pref_order if k in counter] or list(counter.keys())
    sizes = [counter[k] / total for k in keys]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=keys, autopct='%1.1f%%', startangle=90,
            colors=["#66c2a5", "#8da0cb", "#fc8d62"][:len(keys)])
    plt.title("微博舆情情感比例（主帖+评论）")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"情感比例饼状图已保存到 {filename}")

def _clean_text_for_wc(text: str) -> str:
    """简易清洗：去链接/话题/@、控制符、表情符号（基础版），仅为词云可视化用"""
    text = re.sub(r'http\S+', ' ', text)                # 链接
    text = re.sub(r'#([^#]+)#', r' \1 ', text)          # #话题#
    text = re.sub(r'@[\w\-\u4e00-\u9fa5]+', ' ', text)  # @用户
    text = re.sub(r'[\u200b\r\n\t]', ' ', text)         # 零宽空格/换行
    return text

def plot_wordcloud(data, filename="weibo_wordcloud.png"):
    """
    用 第3列(索引2) 的文本内容生成词云
    """
    texts = [row[2] for row in data if len(row) >= 3 and row[2]]
    if not texts:
        print("没有可用于词云的文本，跳过绘制。")
        return

    # 拼接并切词
    raw = " ".join(_clean_text_for_wc(t) for t in texts)
    text_cut = " ".join(jieba.cut(raw))

    stopwords = set(STOPWORDS)
    # 扩充停用词（避免噪声）
    stopwords.update([
        "微博", "转发", "评论", "正文", "链接", "视频", "图片", "网页", "全文",
        "主帖", "评论", "上海", "楼市", "新政", "展开", "收起", "查看", "今天", "大家",
        "一个", "这个", "那个", "还是", "就是", "可以"
    ])

    wc = WordCloud(
        font_path="simhei.ttf", width=1000, height=700,
        background_color="white", stopwords=stopwords
    )
    wc.generate(text_cut)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"词云已保存到 {filename}")

def save_to_excel(data, filename="weibo_posts.xlsx"):
    """
    保持你的保存逻辑不变，只是调用修正后的画图函数
    """
    df = pd.DataFrame(data, columns=["微博ID", "类型", "内容", "情感", "置信度"])
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"数据已保存到 {filename}")
    plot_sentiment_pie(data)     # 现在会用情感列统计
    plot_wordcloud(data)         # 现在会用内容列生成词云

# ---------------------------
# 4) 主流程
# ---------------------------
def run_pipeline(keywords=["上海楼市新政"], max_pages_per_keyword=5):
    all_results = []
    seen_ids = set()

    for kw in keywords:
        print(f"\n=== 抓取关键词: {kw} ===")
        posts = search_weibo_posts(kw, max_pages=max_pages_per_keyword)
        print(f"关键词[{kw}] 共抓到 {len(posts)} 条微博")

        for post in posts:
            if post['mid'] in seen_ids:
                continue
            seen_ids.add(post['mid'])

            # 主帖情感
            label, score = sentiment_score(post['content'])
            all_results.append([post['mid'], "主帖", post['content'], label, score])

            # 抓取评论
            comments = fetch_comments(post['mid'], max_pages=max_pages_per_keyword)
            for com in comments:
                c_label, c_score = sentiment_score(com)
                all_results.append([post['mid'], "评论", com, c_label, c_score])
                time.sleep(random.uniform(0.2, 0.4))

    save_to_excel(all_results)

# ---------------------------
# 运行示例
# ---------------------------
if __name__ == "__main__":
    run_pipeline(keywords=["上海楼市新政", "上海房地产", "沪六条"], max_pages_per_keyword=10)
