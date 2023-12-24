import os

# 허깅페이스 LLM Read Key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# 웹사이트에서 텍스트 가져오기
def get_website_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')  # 웹사이트의 문단(p) 태그를 찾아서 텍스트 추출
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# 웹사이트 URL 설정
website_url = 'https://huggingface.co/docs/transformers/v4.32.0/ko/philosophy'  # 요약할 웹사이트 URL
# 웹사이트 텍스트 가져오기
website_text = get_website_text(website_url)
# Summarization Pipeline 설정
summarizer = pipeline("summarization")
# 웹사이트 문서 요약
summary = summarizer(website_text, max_length=150, min_length=30, do_sample=False)

print(summary[0]['summary_text'])