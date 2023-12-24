import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import PyPDF2
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub

# HuggingFace API 토큰을 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Hugging Face의 transformers 라이브러리에서 모델과 토크나이저를 로드
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# LangChain의 HuggingFaceLLM 인스턴스를 생성
llm = HuggingFaceHub(model=model, tokenizer=tokenizer)
# LangChain을 구성
chain = LLMChain([llm])
# PDF 파일에서 텍스트를 추출하는 함수
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
        return text

# PDF 파일의 경로를 설정
pdf_path = r"C:\Users\user\Desktop\wang2020.pdf"

# PDF에서 텍스트를 추출
pdf_text = extract_text_from_pdf(pdf_path)

# PDF 텍스트를 요약
summary = chain.run(pdf_text, action="summarize")

# 요약 결과를 출력
print(summary)
