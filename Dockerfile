FROM anyscale/ray:2.33.0-slim-py310-cu123

RUN pip install emoji langchain langchain-community \
                sentence-transformers numpy \
                langchain_openai gitpython \
                langchain-huggingface langchainhub einops vllm

RUN pip install "git+https://github.com/flyteorg/flytekit.git@d87c8227663aac246f21229106d9ccf456d6ac68" \
                "git+https://github.com/flyteorg/flytekit.git@d87c8227663aac246f21229106d9ccf456d6ac68#subdirectory=plugins/flytekit-ray"
RUN pip install faiss-gpu

ENV PYTHONPATH=.

COPY . ./