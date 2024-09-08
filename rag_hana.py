from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.hanavector import HanaDB
from hdbcli import dbapi

import os
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

url = os.getenv("url")
port = os.getenv("port")
user = os.getenv("user")
passwd = os.getenv("passwd")

os.environ["AICORE_AUTH_URL"] = os.getenv("AICORE_AUTH_URL")
os.environ["AICORE_CLIENT_ID"] = os.getenv("AICORE_CLIENT_ID")
os.environ["AICORE_CLIENT_SECRET"] = os.getenv("AICORE_CLIENT_SECRET")
os.environ["AICORE_RESOURCE_GROUP"] = os.getenv("AICORE_RESOURCE_GROUP")
os.environ["AICORE_BASE_URL"] = os.getenv("AICORE_BASE_URL")


connection = dbapi.connect(
    address=url,
    port=port,
    user=user,
    password=passwd,
    autocommit=True,
    sslValidationCertificate=False
)

EMBEDDING_DEPLOYMENT_ID = os.getenv("EMBEDDING_DEPLOYMENT_ID")
LLM_DEPLOYMENT_ID = os.getenv("LLM_DEPLOYMENT_ID")

# Defininf which model to use for Chat purpose
chat_llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)

# Load custom Documents
loader = TextLoader('./nexgen_rate_card.txt')

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

texts = text_splitter.split_documents(documents)

print(f"Number of document chunks: {len(texts)}")


# create embeddings for custom documents
embeddings = OpenAIEmbeddings(deployment_id=EMBEDDING_DEPLOYMENT_ID)

db = HanaDB(
    embedding=embeddings, connection=connection, table_name="EMBEDDING_ACG_B3"
)

# Delete already existing documents from the table
db.delete(filter={})

# add the loaded documents chunks
db.add_documents(texts)


# Create a retriever instance to query LLM based on custom documents
retriever = db.as_retriever()

qa = RetrievalQA.from_llm(llm=chat_llm, retriever=retriever)

# streamlit UI Header
st.title("Chat with Nexgen Company")


query = st.text_input("Enter your Query:")


if st.button("Submit"):
    if query:
        response = qa(query)
        st.write("*** Response ***")
        st.write(response['result'])
    else:
        st.write("Please enter your query")