# description

- this is a pure academic research, a practical test for the great stuff from HuggingFace
- no politics agenda is there
- 25 years of multiple speeches and QAs using formal bureaucratic language are ideal for the experiments
- this an average 2000-2025 PTN, not the latest PTN we observe in 2025
- the model is relatively small so all typical small LLMs artifacts and quirks are definitely there
- this is not a dialog model, no context, just a simple single question and answer chat
- consider you are a journalist on the public official conference
- try to ask longer questions with more details
- only russian language is supported, ask the questions in russian

## hugging face

[HuggingFace repository](https://huggingface.co/mazurkin/ptn-gguf)

- 8B model is based on [yandex/YandexGPT-5-Lite-8B-pretrain](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-pretrain)

## example

> A: как лучше всего пропатчить KDE под FreeBSD?

> Q: Это я Вам должен дать ответ или Вы сами должны найти способ ответа на этот вопрос. У нас же есть люди, которые работают в сфере высоких технологий – специалисты по компьютерам и программисты самые разные: «яблочники», пользователи IBM-совместимых компьютеров… Я не могу сказать конкретно сейчас что‑то такое из ряда вон выходящее ни для кого здесь сидящего; если бы мы знали эти вопросы заранее, может быть, тогда можно было подготовить соответствующие ответы. Но это ведь очень сложный раздел деятельности человека вообще с точки зрения поиска решений этих проблем – особенно когда они сформулированы таким специфическим образом. Здесь мне нечего добавить к тому ответу, который был дан моим коллегой только что: всё должно работать так, чтобы государство ничего за собой не тащило вперёд ногами либо назад пятками; все решения принимаются исходя из того интереса населения России (а он является ключевым), которое проживает именно там, где создаётся проблема такого рода при внедрении соответствующих технических средств работы местных властей во взаимодействии со своими гражданами.

# usage

## llama.cpp

Download the files:
- [ptn-8B-q4km.gguf](https://huggingface.co/mazurkin/ptn-GGUF/tree/main)

Run the local command-line chat:
```shell
llama-cli \
    --temp 0.5 \
    --ctx-size 2048 \
    --repeat-penalty 1.2 \
    --repeat-last-n 256 \
    --frequency-penalty 0.5 \
    --presence-penalty 0.5 \
    --model "ptn-8B-q4km.gguf"
```

Or as alternative run the local web-server chat:
```shell
llama-server \
    --temp 0.5 \
    --ctx-size 2048 \
    --repeat-penalty 1.2 \
    --repeat-last-n 256 \
    --frequency-penalty 0.5 \
    --presence-penalty 0.5 \
    --host 127.0.0.1 \
    --port 8080 \
    --model "ptn-8B-q4km.gguf"
```

## ollama

Download the files:
- [ptn-8B-q4km.gguf](https://huggingface.co/mazurkin/ptn-GGUF/tree/main)
- [ptn-8B-q4km.ollama](https://huggingface.co/mazurkin/ptn-GGUF/tree/main)

Create a local model first:
```shell
ollama create ptn-8B-q4km -f ./ptn-8B-q4km.ollama
```

Run the model:
```shell
ollama run ptn-8B-q4km
```

# development

## prerequisites

you must install `conda` first

https://docs.anaconda.com/miniconda/#miniconda-latest-installer-links

## install

```shell
# first, make an isolated Conda environment with Python, Poetry and CUDA inside
$ make env-init-conda

# then install the dependencies with Poetry
$ make env-init-poetry
```

## build

```shell
# train model and convert to GGUF
$ make build
```
