from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
load_dotenv()
urls=[
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]
docs=[WebBaseLoader(urls).load() for url in urls]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=0)
docs_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=docs_splits, collection_name="rag_Chroma", embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")
# retriever= Chroma(collection_name="rag_Chroma", embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db").as_retriever(
#     search_type="similarity", search_kwargs={"k": 3}
#     persistant_directory="./chroma_db"
#     embedding_function=OpenAIEmbeddings()
# ).as_retriever()