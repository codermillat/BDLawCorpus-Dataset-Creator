#!/usr/bin/env python3
"""BDLaws multi-purpose dataset creator

Creates multiple dataset outputs from a folder of JSON act files:
- Finetune instruction-completion JSONL
- RAG passages JSONL
- Simple QA pairs JSONL
- Metadata CSV

Usage (dry-run):
  python projects/dataset-creation/creator.py --input BDLawsActs/acts --output projects/dataset-creation/output --max-files 2 --dry-run
"""
from __future__ import annotations
import argparse
import json
import os
import uuid
from pathlib import Path
import csv
from typing import Iterator, Tuple
import hashlib
import random
import json as _json
import re


# Try to load tiktoken for token-aware splitting; fall back to heuristic
try:
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    def tokenize(text: str):
        return _ENC.encode(text)
except Exception:
    _ENC = None
    def count_tokens(text: str) -> int:
        # fallback heuristic: average 4 chars per token
        return max(1, int(len(text) / 4))
    def tokenize(text: str):
        return text.split()


def load_json_files(input_dir: Path, max_files: int | None = None) -> Iterator[Tuple[str, dict]]:
    files = sorted([p for p in input_dir.glob("*.json")])
    if max_files:
        files = files[:max_files]
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            yield p.name, data
        except Exception as e:
            print(f"WARN: failed to read {p}: {e}")


def extract_text(item: dict) -> str:
    # Try common keys
    for k in ("text", "content", "body", "act_text", "full_text"):
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]

    # Try to concatenate string fields
    parts = []
    for v in item.values():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, list):
            for el in v:
                if isinstance(el, str) and el.strip():
                    parts.append(el.strip())
    return "\n\n".join(parts)


def clean_text(t: str) -> str:
    return " ".join(t.split())


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200, token_aware: bool = False) -> Iterator[Tuple[int, int, str]]:
    """Yield (start_char, end_char, passage). If token_aware=True, prefer token-aware splitting when available.
    Falls back to sentence-accumulation heuristic when tokenizer not available."""
    if not text:
        return
    if token_aware and _ENC is not None:
        tokens = _ENC.encode(text)
        total = len(tokens)
        step = max(1, chunk_size - overlap)
        i = 0
        while i < total:
            j = min(i + chunk_size, total)
            # decode tokens back to text span approximately by decoding token slice
            span = _ENC.decode(tokens[i:j])
            # find char offsets by searching for span in text (best-effort)
            idx = text.find(span)
            if idx != -1:
                yield idx, idx + len(span), span
            else:
                yield 0, len(span), span
            if j == total:
                break
            i += step
        return

    # Fallback: sentence-aware accumulation using punctuation
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    buf = []
    buf_len = 0
    step = max(1, chunk_size - overlap)
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        buf.append(s_clean)
        buf_len += len(s_clean)
        if buf_len >= chunk_size:
            passage = " ".join(buf)
            start = text.find(passage)
            if start == -1:
                start = 0
            yield start, start + len(passage), passage
            # create overlap by keeping last portion
            keep = []
            keep_len = 0
            while buf and keep_len < overlap:
                last = buf.pop()
                keep.insert(0, last)
                keep_len += len(last)
            buf = keep
            buf_len = keep_len
    if buf:
        passage = " ".join(buf)
        start = text.find(passage)
        if start == -1:
            start = 0
        yield start, start + len(passage), passage


def make_finetune_pairs(title: str, text: str, max_len: int = 800) -> list[dict]:
    # Improved finetune templates: include instruction, context, and request for concise structured output
    excerpt = text[:max_len].rstrip()
    pairs = []
    pairs.append({
        "instruction": f"Read the following Act titled '{title}' and provide a concise structured summary (title, year if present, key purpose, and 3 bullet points).",
        "context": excerpt,
        "response": {
            "title": title,
            "summary": excerpt.split('\n')[0] if excerpt else "",
            "bullets": []
        }
    })
    pairs.append({
        "instruction": f"Provide metadata for the Act '{title}' (short title, jurisdiction, estimated length in characters).",
        "context": excerpt,
        "response": {"short_title": title, "jurisdiction": "Bangladesh", "chars": len(text)}
    })
    return pairs


def make_qa_pairs(title: str, text: str, max_answer: int = 400) -> list[dict]:
    ans = text[:max_answer].rstrip()
    qa = []
    qa.append({
        "id": str(uuid.uuid4()),
        "question": f"What is the primary purpose of the Act titled '{title}'?",
        "answer": ans,
        "source": title
    })
    qa.append({
        "id": str(uuid.uuid4()),
        "question": f"Summarize '{title}' in one sentence.",
        "answer": (ans.split('\n')[0] if ans else ""),
        "source": title
    })
    return qa


def write_jsonl(path: Path, iterable):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in iterable:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_metadata_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def create_datasets(input_dir: Path, output_dir: Path, max_files: int | None, chunk_size: int, overlap: int, dry_run: bool = False, use_token_aware: bool = False, dedupe: bool = True):
    finetune = []
    rag_passages = []
    qa_pairs = []
    metadata = []

    seen_hashes: set[str] = set()
    for fname, data in load_json_files(input_dir, max_files=max_files):
        title = data.get("title") or data.get("name") or fname.replace('.json','')
        raw = extract_text(data)
        text = clean_text(raw)
        if not text:
            print(f"WARN: no text extracted from {fname}")
            continue

        # finetune pairs
        ft = make_finetune_pairs(title, text)
        finetune.extend(ft)

        # QA
        qa = make_qa_pairs(title, text)
        qa_pairs.extend(qa)

        # RAG passages (chunked) with dedupe
        count = 0
        for start, end, passage in chunk_text(text, chunk_size=chunk_size, overlap=overlap, token_aware=use_token_aware):
            # normalize and dedupe
            key = hashlib.sha256(passage.strip().lower().encode('utf-8')).hexdigest()
            if dedupe and key in seen_hashes:
                continue
            seen_hashes.add(key)
            pid = str(uuid.uuid4())
            rag_passages.append({
                "id": pid,
                "title": title,
                "act_file": fname,
                "start": start,
                "end": end,
                "text": passage
            })
            count += 1

        metadata.append({
            "act_file": fname,
            "title": title,
            "chars": len(text),
            "passages": count
        })

        if dry_run:
            print(f"DRY: processed {fname}: chars={len(text)}, passages={count}")

    # QC summary
    qc = {
        "finetune_pairs": len(finetune),
        "rag_passages": len(rag_passages),
        "qa_pairs": len(qa_pairs),
        "metadata_rows": len(metadata)
    }

    # Write outputs
    if dry_run:
        out = output_dir
        print("DRY RUN SUMMARY:")
        print(f"  finetune pairs: {len(finetune)}")
        print(f"  rag passages: {len(rag_passages)}")
        print(f"  qa pairs: {len(qa_pairs)}")
        print(f"  metadata rows: {len(metadata)}")
        print("  QC:")
        print(f"    sample finetune: {finetune[:1]}")
        return

    write_jsonl(output_dir / "finetune.jsonl", (o for o in finetune))
    write_jsonl(output_dir / "rag_passages.jsonl", (o for o in rag_passages))
    write_jsonl(output_dir / "qa_pairs.jsonl", (o for o in qa_pairs))
    write_metadata_csv(output_dir / "metadata.csv", metadata)
    # write QC report
    with (output_dir / "qc_report.json").open("w", encoding="utf-8") as fh:
        _json.dump(qc, fh, ensure_ascii=False, indent=2)
    print(f"Wrote datasets to {output_dir} (QC written to qc_report.json)")


def parse_args():
    p = argparse.ArgumentParser(description="BDLaws multi-purpose dataset creator")
    p.add_argument("--input", default="BDLawsActs/acts", help="Input directory with act JSON files")
    p.add_argument("--output", default="projects/dataset-creation/output", help="Output folder")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--token-aware", dest="token_aware", action="store_true", help="Use token-aware splitting when tiktoken is available")
    p.add_argument("--dedupe", dest="dedupe", action="store_true", help="Deduplicate RAG passages")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        print(f"ERROR: input directory {input_dir} does not exist")
        return
    create_datasets(input_dir, output_dir, args.max_files, args.chunk_size, args.overlap, dry_run=args.dry_run, use_token_aware=args.token_aware, dedupe=args.dedupe)


if __name__ == "__main__":
    main()
