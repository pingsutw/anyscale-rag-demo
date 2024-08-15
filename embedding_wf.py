import shutil
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

from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from flytekitplugins.ray.task import AnyscaleConfig

FAISS_INDEX_PATH = "faiss_index"
db_shards = 8

anyscale_config = AnyscaleConfig(compute_config="flyte-rag")
container_image = "pingsutw/rag-demo:latest"


@task(cache_version="1", cache=True, container_image=container_image)
def load_flytekit_repos(chunk_size: int) -> List[Document]:
    url = "https://github.com/flyteorg/flytekit.git"
    name = "flytekit"
    local_path = f"/tmp/{name}"
    shutil.rmtree(local_path, ignore_errors=True)
    Repo.clone_from(url, to_path=local_path)
    docs = GitLoader(
        repo_path=local_path,
        branch="master",
        file_filter=lambda file_path: file_path.endswith(".py"),
    ).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    ).from_language(Language.PYTHON)
    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(documents)} documents from FlyteKit repo")
    return documents


@task(cache_version="1", cache=True, container_image=container_image)
def load_flyte_repos(chunk_size: int) -> List[Document]:
    url = "https://github.com/flyteorg/flyte.git"
    name = "flyte"
    local_path = f"/tmp/{name}"
    shutil.rmtree(local_path, ignore_errors=True)
    Repo.clone_from(url, to_path=local_path)
    docs = GitLoader(
        repo_path=local_path,
        branch="master",
        file_filter=lambda file_path: file_path.endswith(".go"),
    ).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    ).from_language(Language.GO)
    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(documents)} documents from Flyte repo")
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


@task(task_config=anyscale_config, container_image=container_image)
def create_vector_db(
    flytekit: List[Document], flyte: List[Document], slack: List[Document]
) -> FlyteDirectory:
    shards = np.array_split(flytekit + flyte + slack, db_shards)
    futures = [process_shard.remote(shards[i]) for i in range(db_shards)]
    results = ray.get(futures)
    db = results[0]
    for i in range(1, db_shards):
        db.merge_from(results[i])
    db.save_local(FAISS_INDEX_PATH)
    return FAISS_INDEX_PATH


@workflow
def wf() -> FlyteDirectory:
    flytekit = load_flytekit_repos(chunk_size=2000)
    flyte = load_flyte_repos(chunk_size=2000)
    slack = load_slack_data(path="slack.zip", chunk_size=2000)
    return create_vector_db(flytekit, flyte, slack)
