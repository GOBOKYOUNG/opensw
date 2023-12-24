# 필요한 라이브러리 및 모듈 가져오기
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# 문서 처리 관련 모듈 가져오기
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
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

# Map 단계 템플릿 설정
map_template = """다음은 문서 중 일부 내용입니다
{pages}
이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
답변:"""

map_prompt = PromptTemplate.from_template(map_template)

# OpenAI GPT-3.5 모델 초기화
llm = ChatOpenAI(openai_api_key="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", temperature=0, model_name='gpt-3.5-turbo-16k')

# Map 체인 초기화
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce 템플릿 설정
reduce_template = """다음은 요약의 집합입니다:
{doc_summaries}
이것들을 바탕으로 통합된 요약을 만들어 주세요.
답변:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)

# Reduce 체인 초기화
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# 문서 통합 체인 초기화
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,                # Reduce 체인 사용
    document_variable_name="doc_summaries" # 문서 요약 결과 변수 이름
)

# 문서 축소 체인 초기화
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000, # 문서 토큰의 최대 길이
)

# Map-Reduce 체인 초기화
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="pages",        # 문서 페이지 변수 이름
    return_intermediate_steps=False,       # 중간 단계 결과 반환 여부
)

# Map-Reduce 체인 실행
result = map_reduce_chain.run(split_docs)

# 결과 출력
print(result)
