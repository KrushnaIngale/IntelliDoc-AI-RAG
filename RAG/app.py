import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from create_db import create_vector_store

load_dotenv()

st.set_page_config(
    page_title="IntelliDoc AI",
    page_icon="📄",
    layout="wide"
)

# ----------------------------
# SESSION STATE
# ----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# HEADER
# ----------------------------

st.title("📄 IntelliDoc AI")

st.markdown(
    """
Chat with your PDF using
**Mistral AI + ChromaDB + RAG**
"""
)

# ----------------------------
# SIDEBAR
# ----------------------------

with st.sidebar:

    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("Process PDF"):

            with st.spinner("Creating embeddings..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.read())

                    temp_path = temp_file.name

                vector_store = create_vector_store(
                    temp_path
                )

                st.session_state.vector_store = (
                    vector_store
                )

            st.success("Document Ready")

# ----------------------------
# CHAT HISTORY
# ----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# CHAT INPUT
# ----------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    retriever = (
        st.session_state.vector_store
        .as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Use ONLY the provided context.

If answer is not available,
say:

'I could not find the answer in the document.'
"""
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    model = ChatMistralAI(
        model="mistral-small-latest"
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = model.invoke(
        final_prompt
    )

    answer = response.content

    sources = []

    for doc in docs:

        if "page" in doc.metadata:

            page = doc.metadata["page"] + 1

            sources.append(
                f"Page {page}"
            )

    if sources:

        answer += (
            "\n\n---\n**Sources:** "
            + ", ".join(
                list(set(sources))
            )
        )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )