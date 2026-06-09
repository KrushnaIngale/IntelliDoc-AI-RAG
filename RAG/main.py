from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

model=ChatMistralAI(model="mistral-small-latest")

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
    )

retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

prompt=ChatPromptTemplate.from_messages([
    ("system","""You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
    ("human",
            """Context:
{context}

Question:
{question}
""")
])

print("Rag System created")
print("press 0 for exit")

while(True):
    query=input("You: ")

    if query=="0":
        break

    docs=retriever.invoke(query)

    context="\n\n".join([doc.page_content for doc in docs])

    final_prompt=prompt.invoke({"context":context,"question":query})

    response=model.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
    

