import time
import streamlit as st
import openai as client
import re

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🦜",
)

st.title("RAG Assistant")

st.markdown(
    """
Welcome RAG Assistant
Upload a file in the sidebar and ask me anything about it!
"""
)


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def paint_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread):
    messages = get_messages(thread.id)
    for message in messages:
        formatted_text = re.sub(
            r"【[^】]*】", "", message.content[0].text.value.replace("$", "\$")
        )
        if message.role == "assistant":
            paint_message(formatted_text, "ai")
        else:
            paint_message(formatted_text, "human")


assistant_id = "asst_zf07X1iJaD2f4k6xX0dIDV8i"

with st.sidebar:
    file = st.file_uploader(
        "Upload a file (pdf, txt, docx)", type=["pdf", "txt", "docx"]
    )

if file:
    query = st.chat_input("Ask me anything about the file.")
    if "thread" in st.session_state:
        paint_history(st.session_state["thread"])
    if query:
        paint_message(query, "human")
        with st.chat_message("ai"):
            if "thread" not in st.session_state:
                with st.spinner("Uploading file..."):
                    file_path = f"./.cache/files/chunks/{file.name}"
                    with open(file_path, "wb") as fo:
                        fo.write(file.read())
                    my_file = client.files.create(
                        file=client.file_from_path(file_path),
                        purpose="assistants",
                    )
                    thread = client.beta.threads.create(
                        messages=[
                            {
                                "role": "user",
                                "content": query,
                                "attachments": [
                                    {
                                        "file_id": my_file.id,
                                        "tools": [
                                            {"type": "file_search"},
                                        ],
                                    }
                                ],
                            }
                        ]
                    )
                    st.session_state["thread"] = thread
            else:
                thread = st.session_state["thread"]
                send_message(thread.id, query)
            with st.spinner("Generating answer..."):
                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant_id,
                )
                while get_run(run.id, thread.id).status in [
                    "queued",
                    "in_progress",
                    "requires_action",
                ]:
                    time.sleep(0.5)
                st.markdown(
                    re.sub(
                        r"【[^】]*】",
                        "",
                        get_messages(thread.id)[-1]
                        .content[0]
                        .text.value.replace("$", "\$"),
                    )
                )