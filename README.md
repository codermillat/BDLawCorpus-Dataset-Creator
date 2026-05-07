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
# Empirical Quality Evaluation for BDLawCorpus-1

This structured evidence is generated to satisfy formal Academic Research criteria ensuring dataset validity across Quantitative, Safety, and Structural dimensions.

## A. Quantitative Statistics (Corpus Size & Distribution)
- **Total RAG Chunks**: 15428 passages generated across the corpus.
- **RAG Passage Distribution**: Average length = `2581.55` characters. Max length = `10934` characters (Fits 4K/8K LLM windows).
- **Instruction-Tuning Scenarios**: 4710 pairs extracted.
- **Instruct Response Length**: Average length = `260.95` characters.

## B. Alignment & Safety Guardrails (Prompt Engineering Evidence)
- **Disclaimer Injection Rate**: 1570 instances (`33.33%` of finetuning pairs) explicitly carry a predefined safety disclaimer (e.g. "আমি একজন এআই লিগ্যাল এডভাইজার...").
- **Adversarial Refusal Embeddings**: 1570 instances (`33.33%`) strictly guide the model to safely refuse or bound unauthorized advice.

## C. Structural Integrity & Provenance (Traceability)
- **Data Pollution (HTML/JS Trace)**: 0 chunks (`0.00%`) contained polluted HTML boundaries (0% indicates perfect script stripping).
- **Provenance Linkage**:
  - **RAG Traceability**: `100.00%` of passages contain mapping.
  - **Instruct Traceability**: `100.00%` of finetuning pairs contain `act_file` mapping.
