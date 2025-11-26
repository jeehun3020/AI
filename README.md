✨ Basic RAG AI Chatbot with Llama 3 + LangChain2

📚 Multi-Document RAG Chatbot (PDF / TXT / CSV / JSON 지원)
<div align="center">
🚀 Hugging Face Space
👉 https://huggingface.co/spaces/jirtor/LangChain2
</div>
<img width="3352" height="1674" alt="image" src="https://github.com/user-attachments/assets/ec4ec30d-e893-433a-b4c2-95082afffeac" />

🌟 프로젝트 소개

LangChain + FAISS + Sentence Transformers + Groq Llama3 기반으로 동작하는
문서 기반 RAG(Chatbot) 입니다.

업로드된 문서를 읽고 의미 기반 임베딩 → 벡터스토어 → Llama3로
문서 기반 답변을 정확하고 빠르게 생성합니다.

🔥 주요 특징
	•	✔ Streamlit UI
	•	✔ PDF / TXT / CSV / JSON 완전 지원
	•	✔ all-mpnet-base-v2 Sentence Embedding
	•	✔ FAISS로 고속 검색
	•	✔ Groq Llama3.1-8B-Instant으로 초고속 응답
	•	✔ 문맥 유지되는 ConversationalRetrievalChain

⸻

🧠 전체 구조 (Architecture)
문서 업로드
      ↓
문서 로딩 (PDF/TXT/CSV/JSON)
      ↓
RecursiveCharacterTextSplitter
      ↓
Embedding (sentence-transformers/all-mpnet-base-v2)
      ↓
FAISS VectorStore
      ↓
Groq Llama3 + ConversationalRetrievalChain
      ↓
문서 기반 답변 생성


⸻

🛠 지원 문서 포맷
<img width="1652" height="420" alt="image" src="https://github.com/user-attachments/assets/fe7ac2c8-eeff-47d1-9ded-af2dd11a5e3a" />
⚙️ 기술 스택
<img width="1646" height="592" alt="image" src="https://github.com/user-attachments/assets/45779bab-5730-4f79-987c-d894471ad7c7" />

🔧 로컬 실행 방법
1) 클론
git clone https://github.com/jeehun3020/AI --branch Langchain2
cd Langchain2
2) 의존성 설치
pip install -r requirements.txt
3) Groq API Key 설정
export GROQ_API_KEY="your_api_key"
4) 실행
streamlit run src/streamlit_app.py
