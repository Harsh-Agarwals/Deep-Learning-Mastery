from langchain_community.document_loaders import TextLoader

loader = TextLoader("doc.txt")
print(loader)

doc = loader.load()
print(doc)