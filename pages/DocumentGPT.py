import streamlit as st
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Document GPT",
    page_icon="😍"
)
st.title("DocumentGPT")

st.markdown("""
Welcome!
            Use this chatbot to ask questions to anb AI about your files
""")

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(temperature=0.1, streaming=True,callbacks=[ChatCallbackHandler()])
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

@st.cache_data(show_spinner=f"Embedding file......")
def embed_file(file):
    # st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,  # Adjust based on your requirements
    #     chunk_overlap=50,  # Overlap between chunks
    #     separators=["\n"]  # Split based on newlines (you can add other separators too)
    # ) if "xlsx" in file.name else CharacterTextSplitter.from_tiktoken_encoder(
    #                                     separator="\n",
    #                                     chunk_size=600,
    #                                     chunk_overlap=100,
    #                                 )
    # loader = UnstructuredFileLoader(f"./.cache/files/{file.name}") if "xlsx" in file.name else UnstructuredExcelLoader(f"./.cache/files/{file.name}")
    is_xlsx = "xlsx" in file.name
    
    if is_xlsx:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust based on your requirements
            chunk_overlap=50,  # Overlap between chunks
            separators=["\n"],  # Split based on newlines (you can add other separators too)
        )
        loader = UnstructuredExcelLoader(file_path)
    else:
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if(save):
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf .xlsx or .docx file", type=["pdf", "txt", "docx", "xlsx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
