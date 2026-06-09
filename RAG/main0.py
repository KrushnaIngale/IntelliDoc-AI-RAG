from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

data = PyPDFLoader("./document_loader/GRU.pdf")
docs=data.load()
model=ChatMistralAI(model="mistral-small-latest")

print("waiting...........infinitely waiting")

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

context = "\n\n".join([doc.page_content for doc in docs])

question=input("enter your question...")

final_prompt=prompt.invoke({"context":context,"question":question})

result=model.invoke("what is the capital of india?")

response=model.invoke(final_prompt)

print(response.content)



