SHELL := /bin/bash
ROOT  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DATE  := $(shell date '+%Y%m%d-%H%M%S')

CONDA_ENV_NAME      = ptn

REMOTE_HOST        ?= pp-ptn
REMOTE_PATH        ?= projects/ptn

LLAMACPP_PATH      ?= $(HOME)/projects/llamacpp

CUDA_DEVICES_TRAIN ?= 0
CUDA_DEVICES_INFER ?= 1

export PYTHONDONTWRITEBYTECODE = 1
export PYTHONUNBUFFERED        = 1
export PYTHONPATH              = $(ROOT)/src

# -----------------------------------------------------------------------------
# default
# -----------------------------------------------------------------------------

.DEFAULT_GOAL = env-shell

# -----------------------------------------------------------------------------
# conda environment
# -----------------------------------------------------------------------------

.PHONY: env-init-conda
env-init-conda:
	@conda create --yes --copy --name "$(CONDA_ENV_NAME)" \
		conda-forge::python=3.12.12 \
		conda-forge::poetry=2.2.1

.PHONY: env-init-poetry
env-init-poetry:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry install --no-root --no-directory

.PHONY: env-update
env-update:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry update

.PHONY: env-list
env-list:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry show --tree

.PHONY: env-remove
env-remove:
	@conda env remove --yes --name "$(CONDA_ENV_NAME)"

.PHONY: env-shell
env-shell:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		bash

.PHONY: env-python
env-python:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		python3

.PHONY: env-info
env-info:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		conda info

# -----------------------------------------------------------------------------
# convert
# -----------------------------------------------------------------------------

.PHONY: convert
convert:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		python3 src/convert.py

# -----------------------------------------------------------------------------
# tune
# -----------------------------------------------------------------------------

.PHONY: tune
tune: export CUDA_VISIBLE_DEVICES=$(CUDA_DEVICES_TRAIN)
tune:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		python3 src/tune.py

.PHONY: gguf
gguf:
	@mkdir --parents "$(ROOT)/target/gguf"

	@$(LLAMACPP_PATH)/bin/convert_hf_to_gguf.sh \
		--outfile "$(ROOT)/target/gguf/model-f16.gguf" \
		--outtype f16 \
		"$(ROOT)/target/model-merged"

	@$(LLAMACPP_PATH)/llamacpp/build/bin/llama-quantize \
		"$(ROOT)/target/gguf/model-f16.gguf" \
		"$(ROOT)/target/gguf/model-q8.gguf" \
		Q8_0

	@$(LLAMACPP_PATH)/llamacpp/build/bin/llama-quantize \
		"$(ROOT)/target/gguf/model-f16.gguf" \
		"$(ROOT)/target/gguf/model-q4km.gguf" \
		Q4_K_M

	@rm "$(ROOT)/target/gguf/model-f16.gguf"

.PHONY: build
build: clean tune gguf

.PHONY: build-remote
build-remote: rsync-push
	@ssh "$(REMOTE_HOST)" -- "cd $(REMOTE_PATH) && make build"

# -----------------------------------------------------------------------------
# chat
# -----------------------------------------------------------------------------

.PHONY: chat-console
chat-console: export CUDA_VISIBLE_DEVICES=$(CUDA_DEVICES_INFER)
chat-console:
	@"$(LLAMACPP_PATH)/llamacpp/build/bin/llama-cli" \
		--temp 0.5 \
		--ctx-size 2048 \
		--repeat-penalty 1.2 \
		--repeat-last-n 256 \
		--frequency-penalty 0.5 \
		--presence-penalty 0.5 \
		--model "target/gguf/model-q8.gguf"

.PHONY: chat-server
chat-server: export CUDA_VISIBLE_DEVICES=$(CUDA_DEVICES_INFER)
chat-server:
	@"$(LLAMACPP_PATH)/llamacpp/build/bin/llama-server" \
		--temp 0.5 \
		--ctx-size 2048 \
		--repeat-penalty 1.2 \
		--repeat-last-n 256 \
		--frequency-penalty 0.5 \
		--presence-penalty 0.5 \
		--host 0.0.0.0 \
		--port 18080 \
		--model "target/gguf/model-q8.gguf"

.PHONY: chat-remote
chat-remote: rsync-push
	@ssh "$(REMOTE_HOST)" -- "cd $(REMOTE_PATH) && make chat-console"

# -----------------------------------------------------------------------------
# forward
# -----------------------------------------------------------------------------

.PHONY: forward
forward:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		python3 src/forward.py run --port 18080

# -----------------------------------------------------------------------------
# clean
# -----------------------------------------------------------------------------

.PHONY: clean-model
clean-model:
	@rm -rfv "$(ROOT)/target/model"
	@rm -rfv "$(ROOT)/target/model-merged"
	@rm -rfv "$(ROOT)/target/gguf"

.PHONY: clean-tensorboard
clean-tensorboard:
	@rm -rfv "$(ROOT)/target/tensorboard"

.PHONY: clean
clean: clean-model clean-tensorboard

# -----------------------------------------------------------------------------
# tools
# -----------------------------------------------------------------------------

.PHONY: tensorboard
tensorboard:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		tensorboard \
			--logdir "$(ROOT)/target/tensorboard/" \
			--load_fast false \
			--host "0.0.0.0" \
			--port "38001"

.PHONY: vmstat
vmstat:
	@vmstat --unit M --timestamp --wide 3 | tee "$(ROOT)/target/vmstat-$(DATE).log"

# -----------------------------------------------------------------------------
# rsync
# -----------------------------------------------------------------------------

.PHONY: rsync-push
rsync-push:
	@rsync -avz \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/cache/*' \
		--exclude='/target/*' \
		--exclude='*.log' \
		--exclude='.ipynb_checkpoints' \
		'$(ROOT)/' \
		'$(REMOTE_HOST):$(REMOTE_PATH)'

.PHONY: rsync-pull
rsync-pull:
	@rsync -avz \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/cache/*' \
		--exclude='/target/*' \
		--exclude='*.log' \
		--exclude='.ipynb_checkpoints' \
		'$(REMOTE_HOST):$(REMOTE_PATH)' \
		'$(ROOT)/'
