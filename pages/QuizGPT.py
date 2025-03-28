from langchain.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json;

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)
    

output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
            )
        ]
    )

question_chain = {"context": format_docs} | question_prompt | llm


formatting_prompt = ChatPromptTemplate.from_messages([
    (
             "system",
             """
     You are a powerful formatting algorithm.
      
     You format exam questions into JSON format.
     Answers with (o) are the correct ones.
      
     Example Input:
 
     Question: What is the color of the ocean?
     Answers: Red|Yellow|Green|Blue(o)
          
     Question: What is the capital or Georgia?
     Answers: Baku|Tbilisi(o)|Manila|Beirut
          
     Question: When was Avatar released?
     Answers: 2007|2001|2009(o)|1998
          
     Question: Who was Julius Caesar?
     Answers: A Roman Emperor(o)|Painter|Actor|Model
     
      
     Example Output:
      
     ```json
     {{ "questions": [
             {{
                 "question": "What is the color of the ocean?",
                 "answers": [
                         {{
                             "answer": "Red",
                             "correct": false
                         }},
                         {{
                             "answer": "Yellow",
                             "correct": false
                         }},
                         {{
                             "answer": "Green",
                             "correct": false
                         }},
                         {{
                             "answer": "Blue",
                             "correct": true
                         }},
                 ]
             }},
                         {{
                 "question": "What is the capital or Georgia?",
                 "answers": [
                         {{
                             "answer": "Baku",
                             "correct": false
                         }},
                         {{
                             "answer": "Tbilisi",
                             "correct": true
                         }},
                         {{
                             "answer": "Manila",
                             "correct": false
                         }},
                         {{
                             "answer": "Beirut",
                             "correct": false
                         }},
                 ]
             }},
                         {{
                 "question": "When was Avatar released?",
                 "answers": [
                         {{
                             "answer": "2007",
                             "correct": false
                         }},
                         {{
                             "answer": "2001",
                             "correct": false
                         }},
                         {{
                             "answer": "2009",
                             "correct": true
                         }},
                         {{
                             "answer": "1998",
                             "correct": false
                         }},
                 ]
             }},
             {{
                 "question": "Who was Julius Caesar?",
                 "answers": [
                         {{
                             "answer": "A Roman Emperor",
                             "correct": true
                         }},
                         {{
                             "answer": "Painter",
                             "correct": false
                         }},
                         {{
                             "answer": "Actor",
                             "correct": false
                         }},
                         {{
                             "answer": "Model",
                             "correct": false
                         }},
                 ]
             }}
         ]
      }}
     ```
     Your turn!
 
     Questions: {context}
 
 """,
         )
])

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

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
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": question_chain}| formatting_chain | output_parser
    response = chain.invoke(_docs)
    return response

@st.cache_data(show_spinner="Making quiz...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs=wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    

    start = st.button("Generate Quiz")

    if start:
        # question_response = question_chain.invoke(docs)
        # st.write(question_response.content)
        # formatting_response = formatting_chain.invoke({"context": question_response.content}) 
        # st.write(formatting_response.content)
        response = run_quiz_chain(docs, topic if topic else file.name)
        st.write(response)
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()