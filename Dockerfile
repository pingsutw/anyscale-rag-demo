FROM anyscale/ray:2.33.0-slim-py310-cu123

RUN pip install emoji langchain langchain-community \
                sentence-transformers numpy \
                langchain_openai gitpython \
                langchain-huggingface langchainhub einops vllm

RUN pip install "git+https://github.com/flyteorg/flytekit.git@d7e684202dc54a6e8f929e19aeda0053aea130ae" \
                "git+https://github.com/flyteorg/flytekit.git@d7e684202dc54a6e8f929e19aeda0053aea130ae#subdirectory=plugins/flytekit-ray"
RUN pip install faiss-gpu

ENV PYTHONPATH=.

COPY . ./