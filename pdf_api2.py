# 필요한 라이브러리 및 모듈 가져오기
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# 문서 처리 관련 모듈 가져오기
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# PDF 파일 로더 초기화
loader = PyPDFLoader("/Users/user/Desktop/wang2020.pdf")

# 문서 로드
document = loader.load()

# 텍스트 분할 설정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n\n",  # 문장 분리 구분자
    chunk_size=3000,    # 각 청크의 최대 길이
    chunk_overlap=500,  # 청크 간의 겹치는 부분 크기
)

# 문서 분할
split_docs = text_splitter.split_documents(document)

# OpenAI GPT-3.5 모델 초기화
llm = ChatOpenAI(openai_api_key="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", temperature=0, model_name='gpt-3.5-turbo-16k')

# 사용자의 질문 설정
user_question = "What's the most common word in this document?"

# 사용자의 질문과 문서를 합친 템플릿 생성
qa_template = f"""다음은 사용자의 질문입니다:
{user_question}
{user_question}에 대한 답변을 생성해 주세요.
답변:"""

qa_prompt = PromptTemplate.from_template(qa_template)

# 질의응답 체인 초기화
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# 사용자 질문에 대한 답변 생성
answer = qa_chain.run(prompt=user_question, max_tokens=100)  # 최대 토큰 수 설정

# 생성된 답변 출력
print(answer)
