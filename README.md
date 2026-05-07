# BDLawCorpus Dataset Creator

This repository houses the standalone Python pipeline designed to normalize, deduplicate, and compile structural RAG and Finetuning datasets from raw JSON scraping artifacts collected by the [BDLawCorpus Scraper](https://github.com/codermillat/BDLawCorpus).

## Corresponding Resources
- **Raw Scraper (Chrome Extension)**: [codermillat/BDLawCorpus](https://github.com/codermillat/BDLawCorpus)
- **Output HF Dataset**: [millat/BDLawCorpus-Dataset-V1](https://huggingface.co/datasets/millat/BDLawCorpus-Dataset-V1)

## Pipeline Features
1. `creator_safe.py`: Extracts and normalizes data, removing HTML/metadata footprint.
2. Formats 15.4k+ semantic chunks for Vector Embeddings mapping (`rag_passages.jsonl`).
3. Synthesizes safe formatting instructions acting as a "Legal Advisor (লিয়্যাল এডভাইজার)" (`finetune.jsonl`).
4. Generates an empirical report of RAG bounds and safety guardrails.

## Setup
```bash
pip install -r requirements.txt
python creator_safe.py
```
