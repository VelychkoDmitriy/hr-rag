# One-command run: `make run` (or `python main.py all`). On Windows without make, use the python commands below.
PY ?= python

.PHONY: install run corpus ingest index eval-retrieval eval check ask calibrate

install:          ## dependencies (Python 3.12+)
	$(PY) -m pip install -r requirements.txt

run:              ## the whole pipeline: ingest -> index -> retrieval eval -> answer eval with thresholds
	$(PY) main.py all

corpus:           ## download real documents and generate the company documents into data/raw/
	$(PY) scripts/fetch_corpus.py
	$(PY) scripts/make_company_docs.py

ingest:
	$(PY) main.py ingest

index:
	$(PY) main.py index

eval-retrieval:
	$(PY) main.py eval-retrieval

eval:             ## needs OPENROUTER_API_KEY for questions or prompts not in evals/cache/
	$(PY) main.py eval

check:            ## what CI runs: cached retrieval and LLM answers, no key, no models
	HR_RAG_OFFLINE=1 $(PY) main.py eval-retrieval
	HR_RAG_OFFLINE=1 $(PY) main.py eval --check

ask:              ## make ask Q="Скільки днів відпустки має Senior Developer?"
	$(PY) main.py ask "$(Q)"

calibrate:        ## judge vs human labels in evals/human_labels.csv
	$(PY) main.py calibrate
