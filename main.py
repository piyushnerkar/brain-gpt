import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

app = FastAPI()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY", "")

llm = ChatGroq(model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    If the answer is not in the context, say "I don't have enough information to answer that.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

vector_store = None

class QueryRequest(BaseModel):
    question: str

@app.post("/load-knowledge-base")
def load_knowledge_base():
    global vector_store
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        if not os.path.exists("brain_docs"):
            return {"status": "Error", "message": "Directory 'brain_docs' not found."}

        loader = PyPDFDirectoryLoader("brain_docs")
        docs = loader.load()

        if not docs:
            return {"status": "Error", "message": "No documents found in 'brain_docs'."}

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(final_documents, embeddings)
        vector_store.save_local("faiss_index")

        return {"status": "Knowledge base loaded successfully", "pages": len(docs)}

    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/ask")
def ask_question(request: QueryRequest):
    global vector_store
    if vector_store is None:
        if os.path.exists("faiss_index"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            return {"answer": "Please load the knowledge base first."}

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': request.question})
    return {"answer": response['answer']}