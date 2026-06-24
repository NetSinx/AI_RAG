import os
import bs4
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [load_web_page(url) for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)

doc_splits = text_splitter.split_documents(docs_list)

@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={"normalize_embeddings": True},
        ),
    )
    return vectorstore.as_retriever()

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


retriever_tool = retrieve_blog_posts
retriever_tool.invoke({"query": "types of reward hacking"})

response_model = init_chat_model("groq:llama-3.1-8b-instant", temperature=0)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

input = {"messages": [{"role": "user", "content": "hello!"}]}
generate_query_or_respond(input)["messages"][-1].pretty_print()