import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# 🔹 Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# 🔹 Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_text(text)


# 🔹 Create vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vector_store


# 🔹 Create conversational chain (LOCAL MODEL)
def get_conversational_chain(vector_store):

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # lightweight + stable
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain