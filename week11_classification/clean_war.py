# 載入套件
import re

import roman
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer


with open("war.txt", mode="r", encoding="utf-8") as f:
    content = f.read()

rome_num = [f"^{roman.toRoman(num)}\." for num in range(1, 20)]
rome_num_regex = "|".join(rome_num)

content_df = pd.DataFrame({"text": [content]})

content_df = (
    content_df.assign(sentence=content_df["text"].apply(nltk.sent_tokenize))
    .explode("sentence")
    .drop(["text"], axis=1)
)

# 刪除CHAPTER的列
content_df = content_df[~content_df["sentence"].str.contains("CHAPTER")]

# 刪除長度小於1的句子
content_df = content_df[~content_df["sentence"].str.len() < 1]
content_df = content_df.loc[~(content_df["sentence"] == ""), :]


# 刪除標點符號/數字/換行符號
content_df["sentence"] = content_df["sentence"].str.replace(
    r"[^a-zA-Z\s]", ""
)  # 只留下英文字母和空格(包含換行符號)
content_df["sentence"] = content_df["sentence"].str.replace(r"[\n]", " ")  # 將換行符號替換成空格
# tolower
content_df["sentence"] = content_df["sentence"].str.lower()

# 初始化一個PorterStemmer的物件，並存在porter變數中
porter = PorterStemmer()
content_df["sentence"] = content_df["sentence"].apply(porter.stem)

# 初次使用需要安裝nltk中的停用字資源
nltk.download("stopwords")
# 使用nltk的stop_words
stops = stopwords.words("english")
content_df["sentence"] = content_df["sentence"].apply(
    lambda x: " ".join(x for x in x.split() if x not in stops)
)
content_df.to_csv("raw_data/war_clean.csv", index=False, encoding="utf-8")
