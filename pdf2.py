import os
import PyPDF2
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub 
# 허깅페이스 LLM Read Key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
# HuggingFace Repository ID
repo_id = 'mistralai/Mistral-7B-v0.1'
# 템플릿
template = """Context: {context}
Question: {question}
Answer: """
# 프롬프트 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# HuggingFaceHub 객체 생성
llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.2, 
                  "max_length": 128}
)
# LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)
# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
        return text
# 예시 사용
pdf_path = r"C:\Users\user\Desktop\wang2020.pdf"  # PDF 파일 경로
pdf_text = extract_text_from_pdf(pdf_path)  # PDF에서 텍스트 추출
question = "What is ECA-Net?"  # 질문
# 실행
answer = llm_chain.run(context=pdf_text, question=question)
print(answer)
