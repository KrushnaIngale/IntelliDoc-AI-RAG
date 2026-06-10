import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from create_db import create_vector_store

load_dotenv()

st.set_page_config(
    page_title="IntelliDoc AI",
    page_icon="📄"
)

st.title("📄 IntelliDoc AI Book Assistant")
st.write("Upload a PDF and chat with it")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.button("Process PDF"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        with st.spinner("Processing PDF..."):

            st.session_state.vector_store = (
                create_vector_store(temp_path)
            )

        st.success("PDF Ready")

question = st.text_input(
    "Ask a question"
)

if question:

    if st.session_state.vector_store is None:

        st.warning(
            "Upload and process a PDF first."
        )

        st.stop()

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
        [doc.page_content for doc in docs]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not present in the context,
say:

I could not find the answer in the document.
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

    st.subheader("Answer")
    st.write(response.content)