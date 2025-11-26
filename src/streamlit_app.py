import streamlit as st
from dotenv import load_dotenv
# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings  # General embeddings from HuggingFace models.
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import LlamaCpp  # For loading transformer models.
# from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
# 텍스트 스플리터
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# 벡터스토어/임베딩/LLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 로더들 (pebblo/pwd 끌려오지 않게 서브모듈로)
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.json_loader import JSONLoader
import tempfile # 임시 파일을 생성하기 위한 라이브러리입니다.
import os
import json
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

# PDF 문서로부터 텍스트를 추출하는 함수입니다.
def get_pdf_text(pdf_docs):
    temp_dir = tempfile.TemporaryDirectory() # 임시 디렉토리를 생성합니다.
    temp_filepath = os.path.join(temp_dir.name, pdf_docs.name) # 임시 파일 경로를 생성합니다.
    with open(temp_filepath, "wb") as f:  # 임시 파일을 바이너리 쓰기 모드로 엽니다.
        f.write(pdf_docs.getvalue()) # PDF 문서의 내용을 임시 파일에 씁니다.
    pdf_loader = PyPDFLoader(temp_filepath) # PyPDFLoader를 사용해 PDF를 로드합니다.
    pdf_doc = pdf_loader.load() # 텍스트를 추출합니다.
    return pdf_doc # 추출한 텍스트를 반환합니다.


def get_text_file(file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, file.name)

    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())

    txt_loader = TextLoader(temp_filepath, encoding="utf-8")
    txt_doc = txt_loader.load()
    return txt_doc


def get_csv_file(file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, file.name)

    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())

    csv_loader = CSVLoader(temp_filepath)
    csv_doc = csv_loader.load()
    return csv_doc

# def get_json_file(docs):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, docs.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(docs.getvalue())
#     json_loader = JSONLoader(temp_filepath,
#                                  jq_schema='.scans[].relationships',
#                                  text_content=False)
#
#     json_doc = json_loader.load()
#     # print('json_doc = ',json_doc)
#     return json_doc

def get_json_file(file) -> list[Document]:
    # Streamlit UploadedFile -> str
    raw = file.getvalue().decode("utf-8", errors="ignore")
    data = json.loads(raw)

    docs = []

    # 예전 jq 경로가 '.scans[].relationships'였다면, 동일한 의미로 파싱:
    # 존재하면 그것만 뽑고, 없으면 통으로 문서화
    def add_doc(x):
        docs.append(Document(page_content=json.dumps(x, ensure_ascii=False)))

    if isinstance(data, dict) and "scans" in data and isinstance(data["scans"], list):
        for s in data["scans"]:
            rels = s.get("relationships", [])
            if isinstance(rels, list) and rels:
                for r in rels:
                    add_doc(r)
        if not docs:  # 그래도 못 뽑았으면 전체를 하나로
            add_doc(data)
    elif isinstance(data, list):
        for item in data:
            add_doc(item)
    else:
        add_doc(data)

    return docs

# 문서들을 처리하여 텍스트 청크로 나누는 함수입니다.
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 청크의 크기를 지정합니다.
        chunk_overlap=200,  # 청크 사이의 중복을 지정합니다.
        length_function=len  # 텍스트의 길이를 측정하는 함수를 지정합니다.
    )

    documents = text_splitter.split_documents(documents)  # 문서들을 청크로 나눕니다.
    return documents  # 나눈 청크를 반환합니다.


# 텍스트 청크들로부터 벡터 스토어를 생성하는 함수입니다.
def get_vectorstore(text_chunks):
    # 원하는 임베딩 모델을 로드합니다.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs={'device': 'cpu'})  # 임베딩 모델을 설정합니다.
    vectorstore = FAISS.from_documents(text_chunks, embeddings)  # FAISS 벡터 스토어를 생성합니다.
    return vectorstore  # 생성된 벡터 스토어를 반환합니다.


def get_conversation_chain(vectorstore):
    # Groq LLM
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.75,      # 필요에 맞게 튜닝
        max_tokens=512        # 컨텍스트 초과 방지용 (필요시 조정)
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return conversation_chain

# 사용자 입력을 처리하는 함수입니다.
def handle_userinput(user_question):
    print('user_question =>  ', user_question)
    # 대화 체인을 사용하여 사용자 질문에 대한 응답을 생성합니다.
    response = st.session_state.conversation({'question': user_question})
    # 대화 기록을 저장합니다.
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Basic_RAG_AI_Chatbot_with_Llama",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Basic_RAG_AI_Chatbot_with_Llama3 :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your Files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process[PDF]"):
            with st.spinner("Processing"):
                # get pdf text
                doc_list = []
                for file in docs:
                    print('file - type : ', file.type)
                    if file.type in ['application/octet-stream', 'application/pdf']:
                        # file is .pdf
                        doc_list.extend(get_pdf_text(file))
                    else:
                        st.error("PDF 파일이 아닙니다.")
                if not doc_list:
                    st.error("처리 가능한 문서를 찾지 못했습니다.")
                    st.stop()

                text_chunks = get_text_chunks(doc_list)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if st.button("Process[TXT]"):
            if not docs:
                st.error("파일을 업로드하세요.")
                st.stop()

            with st.spinner("Processing TXT..."):
                doc_list = []
                for file in docs:
                    if file.type == 'text/plain':
                        doc_list.extend(get_text_file(file))
                    else:
                        st.error("TXT 형식이 아닙니다.")

                if not doc_list:
                    st.error("처리 가능한 TXT 파일이 없습니다.")
                    st.stop()

                chunks = get_text_chunks(doc_list)
                vector = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vector)

        if st.button("Process[CSV]"):
            if not docs:
                st.error("파일을 업로드하세요.")
                st.stop()

            with st.spinner("Processing CSV..."):
                doc_list = []
                for file in docs:
                    if file.type == 'text/csv':
                        doc_list.extend(get_csv_file(file))
                    else:
                        st.error("CSV 형식이 아닙니다.")

                if not doc_list:
                    st.error("처리 가능한 CSV 파일이 없습니다.")
                    st.stop()

                chunks = get_text_chunks(doc_list)
                vector = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vector)

        

        if st.button("Process[JSON]"):
            with st.spinner("Processing"):
                # get txt text
                doc_list = []
                for file in docs:
                    print('file - type : ', file.type)
                    if file.type == 'application/json':
                        # file is .json
                        doc_list.extend(get_json_file(file))
                    else:
                        st.error("JSON 파일이 아닙니다.")
                if not doc_list:
                    st.error("처리 가능한 문서를 찾지 못했습니다.")
                    st.stop()

                text_chunks = get_text_chunks(doc_list)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()