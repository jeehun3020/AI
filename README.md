✨ Basic RAG AI Chatbot with Llama 3 + LangChain2
📚 Multi-Document RAG Chatbot (PDF / TXT / CSV / JSON 지원)
<div align="center">
🚀 Hugging Face Space:
👉 https://huggingface.co/spaces/jirtor/LangChain2
</div>

🌟 프로젝트 소개

LangChain + Groq + Sentence Transformers 기반 RAG(Chatbot) 을 구현한 Space입니다.
사용자가 업로드한 문서(PDF / TXT / CSV / JSON)를 임베딩 후,
Llama-3.1-8B-Instant 모델을 이용해 문서 기반 질문에 정확한 답변을 생성합니다.

✔ Streamlit UI
✔ 다양한 문서 형식 처리
✔ FAISS 벡터스토어 기반 고속 검색
✔ mpnet-base 임베딩 모델 적용
✔ Groq API 기반 초고속 응답 속도

🧠 전체 구조 (Architecture)
사용자 문서 업로드
        ↓
문서 로딩 (PDF / TXT / CSV / JSON)
        ↓
청크 분할 (RecursiveCharacterTextSplitter)
        ↓
문서 임베딩 (sentence-transformers / all-mpnet-base-v2)
        ↓
FAISS Vectorstore 구축
        ↓
Groq Llama3 모델과 ConversationalRetrievalChain
        ↓
문서 기반 답변 생성

🎨 UI 살짝 보기

(여기에 Space 실행 화면 스크린샷 추가하면 간지 폭발)
📄 Process[PDF]
📝 Process[TXT]
📊 Process[CSV]
🧩 Process[JSON]
각 버튼별로 다른 형식의 문서를 처리하여 벡터스토어를 생성합니다.

🛠 지원되는 문서 포맷
포맷
기능
비고
PDF
PyPDFLoader 기반 텍스트 추출
표·그림 제외
TXT
기본 TextLoader
UTF-8 권장
CSV
CSVLoader로 행별 문서화
표 데이터를 QA로 활용
JSON
커스텀 파서로 관계형 구조까지 읽어냄
.scans[].relationships 자동 처리

⚙️ 기술 스택
영역
사용 기술
Framework
LangChain 0.1, Streamlit
Embedding
sentence-transformers / all-mpnet-base-v2
Vector DB
FAISS
LLM
Groq Llama-3.1-8B-Instant
Parsing
PyPDFLoader, TextLoader, CSVLoader
Deployment
Hugging Face Spaces (Docker)


🔧 설치 (Local 실행용)
1. 클론
git clone https://github.com/jeehun3020/AI --branch Langchain2
cd Langchain2
2. 의존성 설치
pip install -r requirements.txt
3. Groq API Key 설정
export GROQ_API_KEY="your_api_key"
4. 실행
streamlit run src/streamlit_app.py

🚀 HuggingFace Spaces에서 사용 방법
1) 문서를 업로드

좌측 사이드바에서 파일 업로드

2) 문서 형식에 맞는 버튼 클릭
	•	Process[PDF]
	•	Process[TXT]
	•	Process[CSV]
	•	Process[JSON]

3) 질문 입력

문서와 관련된 자연어 질문 입력 → Llama3가 문서 기반으로 답변 생성!

📦 requirements.txt
langchain>=0.1.20,<0.2
langchain-community>=0.0.38,<0.1
langchain-text-splitters
langchain-groq>=0.1.5
PyPDF2==3.0.1
faiss-cpu==1.7.4
pypdf==4.2.0
chromadb==0.4.24
tiktoken==0.7.0
streamlit==1.33.0
streamlit-extras==0.4.2
InstructorEmbedding==1.0.1
sentence-transformers==2.5.1
huggingface-hub==0.22.2
python-dotenv==1.0.1
