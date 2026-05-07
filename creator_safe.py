#!/usr/bin/env python3
"""BDLaws enhanced dataset creator

Adds:
- Act interlink detection and amendment/repeal heuristics
- Enriched metadata (links, amendment flags, effective dates, outdated flag)
- Safer finetune templates including disclaimer/evidence examples to reduce hallucinations

Usage (dry-run):
  python projects/dataset-creation/creator_safe.py --input BDLawsActs/acts --output projects/dataset-creation/output_safe --max-files 2 --dry-run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import csv
import hashlib
import uuid
import re
from typing import Iterator, Tuple

# Tokenizer support (optional)
try:
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    def decode_tokens(tok_slice):
        return _ENC.decode(tok_slice)
except Exception:
    _ENC = None
    def count_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))
    def decode_tokens(tok_slice):
        return tok_slice  # not used in fallback


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
    # First try the rich content fields specific to BDLaws metadata schema
    for k in ("content_corrected", "content_normalized", "content_raw"):
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
    
    # Fallback generic names
    for k in ("text", "content", "body", "act_text", "full_text"):
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
            
    # Absolute worse-case fallback (avoiding metadata noise)
    parts = []
    for k, v in item.items():
        if k.startswith("_") or k in ["url", "content_raw_sha256", "temporal_disclaimer", "content_raw_disclaimer"]:
            continue
        if isinstance(v, str) and v.strip() and len(v) > 200: # Only grab reasonably long blocks to avoid schema noise
            parts.append(v.strip())
            
    return "\n\n".join(parts)


def clean_text(t: str) -> str:
    return " ".join(t.split())


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200, token_aware: bool = False):
    if not text:
        return
    if token_aware and _ENC is not None:
        tokens = _ENC.encode(text)
        total = len(tokens)
        step = max(1, chunk_size - overlap)
        i = 0
        while i < total:
            j = min(i + chunk_size, total)
            span = _ENC.decode(tokens[i:j])
            idx = text.find(span)
            if idx != -1:
                yield idx, idx + len(span), span
            else:
                yield 0, len(span), span
            if j == total:
                break
            i += step
        return
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


def build_act_index(input_dir: Path, max_files: int | None = None) -> dict:
    idx: dict = {}
    for fname, data in load_json_files(input_dir, max_files=max_files):
        title = data.get("title_raw") or data.get("title_normalized") or data.get("title") or data.get("name") or fname.replace('.json', '')
        norm = re.sub(r'[^0-9a-zA-Z\u0980-\u09FF]+', ' ', title).strip().lower()
        idx[norm] = {"file": fname, "title": title}
        m = re.search(r'act\s+no\.?\s*(\d+)', title, flags=re.I)
        if m:
            idx[f"act no {m.group(1)}"] = {"file": fname, "title": title}
    return idx


def extract_dates(text: str) -> list:
    dates = []
    for m in re.finditer(r'\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b', text):
        dates.append(m.group(1))
    for m in re.finditer(r'\b(18|19|20)\d{2}\b', text):
        dates.append(m.group(0))
    return sorted(set(dates))

def detect_amendments_and_links(text: str, acts_index: dict) -> dict:
    low = text.lower()
    am_keywords = ['amend', 'amendment', 'amended', 'repeal', 'repealed', 'repeals', 'replaced', 'commenc', 'comes into force', 'effective', 'সংশোধন', 'সংশোধিত', 'রহিত', 'বাতিল']
    mentions_amendment = any(kw in low for kw in am_keywords)
    repealed = bool(re.search(r'\brepeal(ed|s)?\b', low)) or 'রহিত' in low or 'বাতিল' in low
    
    # Heuristic to detect if it was repealed by a specific act
    replaced_by = ""
    # Look for Bengali: "... দ্বারা রহিত করা হইয়াছে" (Repealed by ...)
    bn_match = re.search(r'(.{0,100}দ্বারা রহিত করা হইয়াছে)', text)
    if bn_match:
        replaced_by = bn_match.group(1).strip()
    
    # Look for English: "... Repealed by the ... Act"
    if not replaced_by:
        en_match = re.search(r'([Rr]epealed by\s+[^.\n]+(?:Act|Ordinance)[^.\n]*)', text)
        if en_match:
            replaced_by = en_match.group(1).strip()

    links = []
    for norm, info in acts_index.items():
        if norm and norm in low and info['file'] not in links:
            links.append(info['file'])
    dates = extract_dates(text)
    flags = {
        'links': links,
        'mentions_amendment': mentions_amendment,
        'repealed': repealed,
        'replaced_by': replaced_by,
        'effective_dates': dates,
        'flag_outdated': bool(repealed or (mentions_amendment and len(dates) > 0) or replaced_by)
    }
    return flags

def make_finetune_pairs(title: str, text: str, max_len: int = 800) -> list[dict]:
    excerpt = text[:max_len].rstrip()
    pairs = []
    pairs.append({
        "instruction": f"Read the following Act titled '{title}' and provide a concise structured summary (title, year if present, key purpose, and 3 bullet points). ALWAYS reply in easy, conversational Bangla (সহজ বাংলা) suitable for common/rural citizens of Bangladesh. Act as a helpful legal advisor (লিয়্যাল এডভাইজার), not a court lawyer. ALWAYS include a short evidence list with source file and character offsets for any statutory claim. If asked for advice, start with a clear disclaimer in Bangla: 'আমি একজন এআই লিগ্যাল এডভাইজার, পেশাদার আইনজীবী নই। আইনি তথ্যের উপর ভিত্তি করে আমি আপনাকে পরামর্শ দিচ্ছি।'",
        "context": excerpt,
        "response": {
            "title": title,
            "summary": "আইনের মূল উদ্দেশ্য সহজ বাংলায় এখানে লিখুন।",
            "bullets": [],
            "evidence": []
        }
    })
    pairs.append({
        "instruction": f"Provide metadata for the Act '{title}' (short title, jurisdiction, estimated length in characters). Include detected cross-act links and amendment flags if present.",
        "context": excerpt,
        "response": {"short_title": title, "jurisdiction": "Bangladesh", "chars": len(text), "links": [], "mentions_amendment": False}
    })
    pairs.append({
        "instruction": "If a rural or common citizen asks for specific legal guidance about their situation, respond in easy Bangla (সহজ বাংলা). Respond as a legal advisor helping them understand the law, but start with a disclaimer. Do not give final court rulings, but rather guide them on what the law says and advise them to consult a local lawyer or legal aid office.",
        "context": "",
        "response": {
            "disclaimer": "আমি একজন এআই লিগ্যাল এডভাইজার, পেশাদার আইনজীবী নই। আইনি তথ্যের উপর ভিত্তি করে আমি আপনাকে সাধারণ পরামর্শ দিচ্ছি। (I am an AI legal advisor, not a lawyer. I am providing general advice based on legal info.)",
            "info": "আইনের প্রাসঙ্গিক ধারা সহজ বাংলায় বুঝিয়ে বলুন এবং সোর্স উল্লেখ করুন। (Explain the relevant sections of the law in easy Bangla and cite sources.)",
            "next_steps": "প্রয়োজনে স্থানীয় কোনো আইনজীবী বা লিগ্যাল এইড (Legal Aid) এর সাথে যোগাযোগ করার পরামর্শ দিন।"
        }
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
    # Load all acts first to build index for cross-references
    acts = []
    for fname, data in load_json_files(input_dir, max_files=max_files):
        title = data.get("title_raw") or data.get("title_normalized") or data.get("title") or data.get("name") or fname.replace('.json','')
        raw = extract_text(data)
        text = clean_text(raw)
        acts.append((fname, title, text))

    acts_index = build_act_index(input_dir, max_files=max_files)

    finetune = []
    rag_passages = []
    qa_pairs = []
    metadata = []
    seen_hashes: set[str] = set()

    for fname, title, text in acts:
        if not text:
            print(f"WARN: no text extracted from {fname}")
            continue

        # detect amendments / links
        flags = detect_amendments_and_links(text, acts_index)

        # finetune pairs
        ft = make_finetune_pairs(title, text)
        # attach provenance hint to each finetune pair
        for p in ft:
            p.setdefault('provenance', {})
            p['provenance']['act_file'] = fname
        finetune.extend(ft)

        # QA
        qa = make_qa_pairs(title, text)
        qa.append({
            "id": str(uuid.uuid4()),
            "question": "সম্পূর্ণ ভিন্ন কোনো আইন বা এই প্রসঙ্গের বাইরের কোনো তথ্য কি দিতে পারবেন?",
            "answer": "দুঃখিত, এই বিষয়ে আইনে স্পষ্ট কিছু উল্লেখ নেই বা আমার কাছে তথ্য নেই।",
            "source": "None (Adversarial Refusal)"
        })
        qa_pairs.extend(qa)

        # RAG passages (chunked) with dedupe
        count = 0
        for start, end, passage in chunk_text(text, chunk_size=chunk_size, overlap=overlap, token_aware=use_token_aware):
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
            "passages": count,
            "links": ",".join(flags['links']) if flags.get('links') else "",
            "mentions_amendment": flags.get('mentions_amendment', False),
            "repealed": flags.get('repealed', False),
            "replaced_by": flags.get('replaced_by', ""),
            "effective_dates": ",".join(flags.get('effective_dates', [])),
            "flag_outdated": flags.get('flag_outdated', False)
        })

        if dry_run:
            print(f"DRY: processed {fname}: chars={len(text)}, passages={count}, flags={ {k:v for k,v in metadata[-1].items() if k in ['mentions_amendment','repealed','replaced_by','flag_outdated']} }")

    qc = {
        "finetune_pairs": len(finetune),
        "rag_passages": len(rag_passages),
        "qa_pairs": len(qa_pairs),
        "metadata_rows": len(metadata),
        "failed_acts": len(list(input_dir.parent.joinpath("failed").glob("*"))) if input_dir.parent.joinpath("failed").exists() else 0
    }

    if dry_run:
        print("DRY RUN SUMMARY:")
        print(f"  finetune pairs: {len(finetune)}")
        print(f"  rag passages: {len(rag_passages)}")
        print(f"  qa pairs: {len(qa_pairs)}")
        print(f"  metadata rows: {len(metadata)}")
        print("  QC:")
        print(f"    sample finetune: {finetune[:1]}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "finetune.jsonl", (o for o in finetune))
    write_jsonl(output_dir / "rag_passages.jsonl", (o for o in rag_passages))
    write_jsonl(output_dir / "qa_pairs.jsonl", (o for o in qa_pairs))
    write_metadata_csv(output_dir / "metadata.csv", metadata)
    with (output_dir / "qc_report.json").open("w", encoding="utf-8") as fh:
        json.dump(qc, fh, ensure_ascii=False, indent=2)
    print(f"Wrote datasets to {output_dir} (QC written to qc_report.json)")


def parse_args():
    p = argparse.ArgumentParser(description="BDLaws enhanced dataset creator")
    p.add_argument("--input", default="BDLawsActs/acts", help="Input directory with act JSON files")
    p.add_argument("--output", default="projects/dataset-creation/output_safe", help="Output folder")
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
