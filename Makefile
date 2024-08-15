export REPOSITORY=anyscale-ray-demo


.PHONY: setup
setup: ## Install requirements
	pip install flytekit flytekitplugins-ray emoji langchain langchain-community \
 				sentence-transformers numpy langchain_openai gitpython \
 				langchain-huggingface langchainhub

.PHONY: fmt
fmt:
	pre-commit run ruff --all-files || true
	pre-commit run ruff-format --all-files || true


.PHONY: embedding
embedding:
	pyflyte run embedding_wf.py wf


.PHONY: inference
inference:
	pyflyte -vv run inference_wf.py wf


.PHONY: register
register:
	pyflyte register embedding_wf.py inference_wf.py