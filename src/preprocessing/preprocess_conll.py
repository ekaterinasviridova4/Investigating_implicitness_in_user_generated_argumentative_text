#!/usr/bin/env python3
import argparse, json, os, re, random
from pathlib import Path

def parse_conll_file(conll_file_path):
    """Parse blank-line separated texts and DROP all O tokens."""
    data, tokens, ner_tags = [], [], []
    # Tag2id dictionary for Implicit and Explicit tags
    tag2id = {"O":0, "B-Implicit":1, "I-Implicit":2, "B-Explicit":3, "I-Explicit":4}
    with open(conll_file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line == "":
                if tokens:
                    data.append({"id": str(len(data)), "tokens": tokens, "ner_tags": ner_tags})
                tokens, ner_tags = [], []
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"Skipping malformed line {i}: {line}")
                continue
            token, _, tag = parts
            if tag == "O":  # drop O tokens
                continue
            if tag not in tag2id:
                print(f"Unknown tag '{tag}' at line {i}, skipping.")
                continue
            tokens.append(token)
            ner_tags.append(tag2id[tag])
    if tokens:
        data.append({"id": str(len(data)), "tokens": tokens, "ner_tags": ner_tags})
    return data

def reconstruct_sentence(tokens):
    s = ""
    for tok in tokens:
        if re.match(r'^[.,!?;:\'\")\]]$', tok):
            s += tok
        elif tok in ['(', '[', '"', "'"]:
            s += " " + tok
        else:
            s += " " + tok
    return s.strip()

def split_sentences_from_tags(tokens, tags):
    """Split by B- tags and type switches (Implicit <-> Explicit)."""
    spans, cur_toks, cur_tags, prev_type = [], [], [], None
    # Type function for Implicit/Explicit scheme
    def ttype(tid):
        return "Implicit" if tid in (1,2) else ("Explicit" if tid in (3,4) else "O")
    for tok, tag in zip(tokens, tags):
        ctype = ttype(tag)
        # B- tag detection for Implicit/Explicit scheme
        if (tag in (1,3)) or (prev_type is not None and ctype != prev_type):
            if cur_toks:
                spans.append((cur_toks, cur_tags))
            cur_toks, cur_tags = [tok], [tag]
        else:
            cur_toks.append(tok); cur_tags.append(tag)
        prev_type = ctype
    if cur_toks:
        spans.append((cur_toks, cur_tags))
    return spans

def wrap_text(tokens, tags):
    if not tokens:
        return ""
    parts = []
    for span_toks, span_tags in split_sentences_from_tags(tokens, tags):
        # Implicit/Explicit labeling
        label = "Implicit" if span_tags[0] in (1,2) else ("Explicit" if span_tags[0] in (3,4) else None)
        
        sent = reconstruct_sentence(span_toks)
        parts.append(f"<{label}> {sent} </{label}>" if label else sent)
    return " ".join(parts).strip()

def write_jsonl(items, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            # Tagged version (with <Implicit>/<Explicit>)
            tagged_text = wrap_text(it["tokens"], it["ner_tags"])
            if not tagged_text.strip():
                continue

            # Untagged version (just the plain sentence reconstruction)
            untagged_text = reconstruct_sentence(it["tokens"])

            obj = {
                "input": untagged_text,   # raw text
                "output": tagged_text     # gold annotated text
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n

def split_indices(n, train_ratio=0.80, dev_ratio=0.10, test_ratio=0.10, seed=0):
    """Flooring split with deterministic local RNG."""
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    train_size = int(train_ratio * n)
    dev_size   = int(dev_ratio * n)
    test_size  = n - train_size - dev_size  # remainder
    train_idx = idx[:train_size]
    dev_idx   = idx[train_size:train_size + dev_size]
    test_idx  = idx[train_size + dev_size:]
    return train_idx, dev_idx, test_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".conll path")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--splits", nargs=3, type=float, default=[0.80, 0.10, 0.10], help="train dev test ratios")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    data = parse_conll_file(args.input)
    # Ensure stable base order before shuffling indices
    data = sorted(data, key=lambda d: int(d["id"]))
    n = len(data)
    print(f"Parsed {n} texts after dropping O tokens.")
    print(f"Using ratios={args.splits} with seed={args.seed}")

    tr_idx, dv_idx, te_idx = split_indices(n, *args.splits, seed=args.seed)
    train = [data[i] for i in tr_idx]
    dev   = [data[i] for i in dv_idx]
    test  = [data[i] for i in te_idx]

    n_tr = write_jsonl(train, os.path.join(args.outdir, "train.jsonl"))
    n_dv = write_jsonl(dev,   os.path.join(args.outdir, "dev.jsonl"))
    n_te = write_jsonl(test,  os.path.join(args.outdir, "test.jsonl"))
    print(f"Wrote: train={n_tr}, dev={n_dv}, test={n_te} to {args.outdir}")

if __name__ == "__main__":
    main()