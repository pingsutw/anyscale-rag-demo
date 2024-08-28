import typing
from typing import List

from git import Repo
from langchain_community.document_loaders import GitLoader, SlackDirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import ray
import numpy as np
from langchain_core.vectorstores import VST
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
    CharacterTextSplitter,
)

from flytekit import task, workflow, FlyteContext
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from flytekitplugins.ray.task import AnyscaleConfig

FAISS_INDEX_PATH = "faiss_index"
db_shards = 8

anyscale_config = AnyscaleConfig(compute_config="flyte-rag")
container_image = "pingsutw/rag-demo:latest"


@task(cache_version="1", cache=True, container_image=container_image)
def download_data() -> typing.Tuple[FlyteDirectory, FlyteDirectory]:
    ctx = FlyteContext.current_context()

    url = "https://github.com/flyteorg/flyte.git"
    flyte_repo = ctx.file_access.get_random_local_directory()
    Repo.clone_from(url, to_path=flyte_repo)

    url = "https://github.com/flyteorg/flytekit.git"
    flytekit_repo = ctx.file_access.get_random_local_directory()
    Repo.clone_from(url, to_path=flytekit_repo)

    return flyte_repo, flytekit_repo


@task(cache_version="1", cache=True, container_image=container_image)
def load_flytekit_code(repo: FlyteDirectory, chunk_size: int) -> List[Document]:
    docs = GitLoader(
        repo_path=repo,
        branch="master",
        file_filter=lambda file_path: file_path.endswith(".py"),
    ).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    ).from_language(Language.PYTHON)
    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(documents)} documents from FlyteKit repository")
    return documents


@task(cache_version="1", cache=True, container_image=container_image)
def load_flyte_code(repo: FlyteDirectory, chunk_size: int) -> List[Document]:
    docs = GitLoader(
        repo_path=repo,
        branch="master",
        file_filter=lambda file_path: file_path.endswith(".go"),
    ).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    ).from_language(Language.GO)
    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(documents)} documents from Flyte repository")
    return documents


@task(cache_version="1", cache=True, container_image=container_image)
def load_flyte_document(repo: FlyteDirectory, chunk_size: int) -> List[Document]:
    docs = GitLoader(
        repo_path=repo,
        branch="master",
        file_filter=lambda file_path: file_path.endswith(".rst"),
    ).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    ).from_language(Language.RST)
    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(documents)} documents from Flyte documents")
    return documents


@task(cache_version="1", cache=True, container_image=container_image)
def load_slack_data(path: FlyteFile, chunk_size: int) -> List[Document]:
    loader = SlackDirectoryLoader(zip_path=path)
    raw_documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Loaded {len(documents)} documents from {path}")
    return documents


@ray.remote(num_gpus=0)
def process_shard(shard) -> VST:
    print(f"Starting process_shard of {len(shard)} chunks.")
    st = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    result = FAISS.from_documents(shard, embeddings)
    et = time.time() - st
    print(f"Shard completed in {et} seconds.")
    return result


class EmbedChunks:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def __call__(self, batch: typing.Dict[str, np.ndarray]) -> typing.Dict[str, list]:
        results = FAISS.from_documents(batch["data"], self.embedding_model)
        return {"embeddings": [results]}


@task(task_config=anyscale_config, container_image=container_image)
def embedding_generation(
    flytekit_code: List[Document],
    flyte_code: List[Document],
    flyte_document: List[Document],
    slack: List[Document],
) -> FlyteDirectory:
    docs = flytekit_code + flyte_code + flyte_document + slack
    shards = np.array_split(docs, db_shards)
    ds = ray.data.from_numpy(shards)
    res = ds.map_batches(
        EmbedChunks,
        num_gpus=0,
        batch_size=1000,
        concurrency=2,
    ).take_all()

    db = res[0]["embeddings"]
    for i in range(1, len(res)):
        db.merge_from(res[i]["embeddings"])
    db.save_local(FAISS_INDEX_PATH)
    return FAISS_INDEX_PATH


@workflow
def wf() -> FlyteDirectory:
    flyte_repo, flytekit_repo = download_data()

    flytekit_code = load_flytekit_code(repo=flytekit_repo, chunk_size=2000)
    flyte_code = load_flyte_code(repo=flyte_repo, chunk_size=2000)
    flyte_document = load_flyte_document(repo=flyte_repo, chunk_size=2000)
    slack = load_slack_data(path="slack.zip", chunk_size=2000)
    return embedding_generation(flytekit_code, flyte_code, flyte_document, slack)
