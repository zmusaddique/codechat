import os
import re
import asyncio
import chromadb
import textwrap
import streamlit as st
from typing import List
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from llama_index import download_loader, VectorStoreIndex, ServiceContext, PromptTemplate
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.storage.storage_context import StorageContext
from llama_hub.github_repo import GithubRepositoryReader, GithubClient

# get retrievers
from llama_index import QueryBundle
from llama_index.retrievers import BM25Retriever, BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.schema import NodeWithScore
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores import ChromaVectorStore


load_dotenv()

llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",  # replace with your model name
    context_window=2048,  # to use refine
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  # replace with your HuggingFace token
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    chunk_size=1024,
    chunk_overlap=64,
)


def parse_github_url(url):
    pattern = r"https:\/\/github\.com\/([^/]+)\/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)


def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def initialize_github_client():
    github_token = os.getenv("GITHUB_TOKEN")
    return GithubClient(github_token)

query_str = ""

query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

results_dict = {}


def generate_queries(llm, query_str: str, num_queries: int = 4):
    query_gen_prompt = PromptTemplate(query_gen_prompt_str)
    fmt_prompt = query_gen_prompt.format(num_queries=num_queries - 1, query=query_str)
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries


def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = asyncio.run(tqdm.gather(*tasks))

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict


def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(llm, query_str, num_queries=4)
        results = run_queries(queries, self._retrievers)
        final_results = fuse_results(
            results, similarity_top_k=self._similarity_top_k
        )

        return final_results


def validate_token_presence():
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not huggingfacehub_api_token:
        raise EnvironmentError(
            "HuggingFaceHub API key not found in environment variables"
        )

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("Github token not found in environment variables")

@st.cache_resource(show_spinner=False)
def create_docs_and_nodes(github_url):
    validate_token_presence()
    github_client = initialize_github_client()
    download_loader("GithubRepositoryReader")
    print("loader loaded")

    while True:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            loader = GithubRepositoryReader(
                github_client,
                owner=owner,
                repo=repo,
                filter_file_extensions=(
                    [".py", ".js", ".ts", ".md", ".ipynb"],
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
                verbose=False,
                concurrent_requests=20,
            )
            print(f"Loading {repo} repository by {owner}")
            docs = loader.load_data(branch="main")

            print("Documents uploaded: ")
            for doc in docs:
                print(doc.metadata)
            service_context = ServiceContext.from_defaults(
                llm=llm, embed_model=embed_model
            )
            nodes = service_context.node_parser.get_nodes_from_documents(docs)
            break  # Exit the loop once the valid URL is processed
        else:
            print("Invalid Github URL. Please try again.")
    return docs, nodes


def create_vector_store(docs, nodes):
    # -------------Create vector store and upload data---------------

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection("codechat")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    vector_index = VectorStoreIndex(
        docs,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    return vector_index


async def create_query_engine(vector_index, nodes):
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)
    # bm25 retriever
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

    global query_str
    query_str = "What is the repository about and what is the tech stack?"
    queries = generate_queries(llm, query_str, num_queries=4)

    global results_dict
    results_dict = run_queries(queries, [vector_retriever, bm25_retriever])

    fusion_retriever = FusionRetriever(
        llm, [vector_retriever, bm25_retriever], similarity_top_k=2
    )

    response_synthesizer = get_response_synthesizer(service_context=service_context)
    query_engine = RetrieverQueryEngine(
        fusion_retriever,
        response_synthesizer=response_synthesizer,
    )
    return query_engine

async def main(github_url):
    docs, nodes = create_docs_and_nodes(github_url)
    print("Uploading to vector store...")

    # -------------Create vector store and upload data---------------

    vector_index = create_vector_store(docs, nodes)

    print("Fusion starting")
    query_engine = await create_query_engine(vector_index, nodes)

    response = query_engine.query(query_str)
    print(str(response))

    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Exiting, Thanks for chatting!")
            break
        print("=" * 50)
        print(f"Your question: {user_question}")

        response = query_engine.query(user_question)
        print(f"Answer: {textwrap.fill(str(response), 100)} \n")


if __name__ == "__main__":
    asyncio.run(main("https://github.com/zmusaddique/chatbot-restaurant"))