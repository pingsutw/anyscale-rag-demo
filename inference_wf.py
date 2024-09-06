import json
import os
import ray
import typing
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import requests
from langchain_community.llms.vllm import VLLM
from markdown_it import MarkdownIt
import flytekit
from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.ray.task import AnyscaleConfig

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # noqa: F401


"""
To run this workflow locally, you need to have the following environment variables set:

    export ANYSCALE_API_KEY=...
    export GITHUB_PERSONAL_ACCESS_TOKEN=...
    export GITHUB_BOT_ACCESS_TOKEN=...
"""

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
anyscale_config = AnyscaleConfig(compute_config="flyte-rag")
container_image = "pingsutw/rag-demo:latest"


@task(cache_version="1", cache=True, container_image=container_image)
def load_github_issues() -> typing.List[Document]:
    since = datetime.now() - timedelta(days=1)

    loader = GitHubIssuesLoader(
        repo="flyteorg/flyte",
        access_token=GITHUB_PERSONAL_ACCESS_TOKEN,
        sort="created",
        since=str(since.strftime("%Y-%m-%dT%H:%M:%SZ")),
        creator=None,
        include_prs=False,
    )
    data = loader.load()
    return data


class LlamaPredictor:
    def __init__(self, vector_database: FlyteDirectory):
        vector_database.download()

        db = FAISS.load_local(
            vector_database,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
            allow_dangerous_deserialization=True,
        )
        retriever = db.as_retriever()
        # llm = ChatOpenAI(model="gpt-4o")
        # https://python.langchain.com/v0.2/docs/integrations/llms/vllm/
        llm = VLLM(
            model="meta-llama/Meta-Llama-3.1-70B",
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=512,
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        response_schemas = [
            ResponseSchema(name="answer", description="answer to the user's question"),
            ResponseSchema(
                name="source",
                description="source used to answer the user's question, should be a website.",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template="You are a Flyte expert helping people resolve the github issues. "
            "Use the following pieces of retrieved context to answer the questions "
            "and propose code changes as best as possible.\n{context}"
            "\n{format_instructions}\n{question}",
            input_variables=["question", "context"],
            partial_variables={"format_instructions": format_instructions},
        )

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        questions = batch["data"]
        batch["output"] = [self.rag_chain.invoke(question) for question in questions]
        return batch


@task(container_image=container_image, task_config=anyscale_config, enable_deck=True)
def batch_inference(issues: typing.List[Document], vector_database: FlyteDirectory):
    questions = [issue.page_content for issue in issues]
    ds = ray.data.from_numpy(np.asarray(questions))
    predictions = ds.map_batches(
        LlamaPredictor,
        num_gpus=2,
        batch_size=256,
        concurrency=2,
        fn_constructor_kwargs={"vector_database": vector_database},
    )

    answers = predictions.take_all()

    for i in range(len(answers)):
        issue_number = issues[i].metadata["url"].split("/")[-1]
        if answers[i]["output"]["source"]:
            source = "https://github.com/flyteorg/flyte"
        else:
            source = answers[i]["output"]["source"]
        response = f"""
        {MarkdownIt().render(answers[i]["output"]["answer"])}
        Sources:
        <a href={source}>Flyte Documentations</a>
        <br>
        This is an AI-generated response and your feedback is appreciated! Please leave a 👍 if this is helpful and 👎 if it is not.
        """
        flytekit.Deck(f"Issue {issue_number}", response)

        leave_comments(issue_number, answers[i]["output"]["answer"])


def leave_comments(issue_number: str, response: str):
    headers = {
        "accept": "application/vnd.github+json",
        "Authorization": "Bearer " + os.getenv("GITHUB_BOT_ACCESS_TOKEN"),
    }
    requests.post(
        url=f"https://api.github.com/repos/flyteorg/flyte/issues/{issue_number}/comments",
        data=json.dumps({"body": response}),
        headers=headers,
    )


@workflow
def wf(vector_database: FlyteDirectory = "faiss_index"):
    issues = load_github_issues()
    batch_inference(issues=issues, vector_database=vector_database)
