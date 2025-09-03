# 读取之前爬取好的weibo_posts.xlsx，将评论交给deepseek进行情感分类。

import pandas as pd
import time
import random
import jieba
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
from openai import OpenAI

# ============ 配置 ============
API_KEY = "sk-f1b6bbfe4e6f4a92b6cba7b19bb6c678"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

INPUT_FILE = "weibo_posts.xlsx"
OUTPUT_FILE = "weibo_posts_labeled.xlsx"

# ============ Step 1: 读取Excel ============
df = pd.read_excel(INPUT_FILE)

# 确保列名一致
# 微博ID	类型	内容	情感	置信度
assert {"微博ID", "类型", "内容"}.issubset(df.columns)

# ============ Step 2: 调用 DeepSeek ============
def classify_sentiment(main_post, comment):
    """
    调用 deepseek API，判断评论相对主帖的情感。
    返回: "正面" / "负面" / "中立"
    """
    prompt = f"""
主帖内容: {main_post}
评论内容: {comment}

请判断该评论相对于主帖的态度情感，只能输出以下三种类别之一：
- 正面
- 负面
- 中立

请只输出类别，不要解释。
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Chinese text analysis."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        label = response.choices[0].message.content.strip()
        if label not in ["正面", "负面", "中立"]:
            label = "中立"  # fallback
        return label
    except Exception as e:
        print(f"API error: {e}")
        return "中立"

# ============ Step 3: 遍历主帖和评论 ============
results = []
for mid, group in df.groupby("微博ID"):
    main_post = group[group["类型"] == "主帖"]["内容"].values[0]

    for _, row in group.iterrows():
        if row["类型"] == "评论":
            comment = row["内容"]
            label = classify_sentiment(main_post, comment)
            results.append([mid, "评论", comment, label])
            print(f"[{mid}] 评论: {comment[:20]}... → {label}")
            time.sleep(random.uniform(0.5, 1.0))  # 防止请求过快

# 保存结果到Excel
out_df = pd.DataFrame(results, columns=["微博ID", "类型", "内容", "情感"])
out_df["微博ID"] = out_df["微博ID"].astype(str)
out_df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"已保存结果到 {OUTPUT_FILE}")

# ============ Step 4: 绘图 ============
# 评论文本汇总
all_comments = out_df["内容"].astype(str).tolist()
text = " ".join(all_comments)
words = jieba.cut(text)
text_cut = " ".join(words)

stopwords = set(STOPWORDS)
stopwords.update(["微博", "转发", "评论", "上海", "楼市", "新政"])

# 词云
wc = WordCloud(font_path="simhei.ttf", width=800, height=600,
               background_color="white", stopwords=stopwords).generate(text_cut)
plt.figure(figsize=(10,8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("comments_wordcloud.png", dpi=300)
plt.close()
print("词云已保存到 comments_wordcloud.png")

# 饼状图
counter = Counter(out_df["情感"])
total = sum(counter.values())
sizes = [v/total for v in counter.values()]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=counter.keys(), autopct="%1.1f%%", startangle=90,
        colors=["#66c2a5","#fc8d62","#8da0cb"])
plt.title("评论情感比例")
plt.savefig("comments_sentiment_pie.png", dpi=300)
plt.close()
print("评论情感饼状图已保存到 comments_sentiment_pie.png")
