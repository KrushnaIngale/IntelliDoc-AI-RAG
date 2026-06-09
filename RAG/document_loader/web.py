from langchain_community.document_loaders import WebBaseLoader

url="https://www.geeksforgeeks.org/dsa/string-data-structure/"

data = WebBaseLoader(url)

docs=data.load()

print(docs[0].page_content)
