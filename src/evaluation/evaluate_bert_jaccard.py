# evaluate with Jaccard and BERT-based matching, with detailed diagnostics and reports

import os
import re
import csv
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


TAG_PATTERN = re.compile(
    r"<\s*(Implicit|Explicit)\s*>\s*(.*?)\s*(?=<\s*/\s*\1\s*>|<\s*(?:Implicit|Explicit)\s*>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def normalize_space(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(s.split()).strip()

def tokenize(s: str) -> List[str]:
    s = normalize_space(s.lower())
    return TOKEN_PATTERN.findall(s)

def extract_spans_from_text(tagged_text: str) -> List[Dict]:
    """
    Returns list of dicts: {label: 'Implicit'|'Explicit', text: raw_span, tokens: [..]}
    Robust to extra whitespace/newlines and tag case.
    """
    spans = []
    for m in TAG_PATTERN.finditer(tagged_text):
        label = m.group(1).capitalize()  # normalize to 'Implicit'/'Explicit'
        span_text = m.group(2).strip()
        spans.append({
            "label": label,
            "text": span_text,
            "tokens": tokenize(span_text),
        })
    return spans

# Jaccard Matching

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def match_doc_spans_jaccard(pred_spans: List[Dict], gold_spans: List[Dict], threshold: float = 0.8):
    """
    Greedy highest-overlap matching per document, label-must-match, thresholded.
    Returns:
      matches: list of tuples (gold_idx, pred_idx, gold_label, pred_label, overlap, gold_text, pred_text)
      unmatched_gold_idx: set of remaining gold indices
      unmatched_pred_idx: set of remaining pred indices
    """
    candidates = []
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            if g["label"] != p["label"]:
                continue
            overlap = jaccard(g["tokens"], p["tokens"])
            if overlap >= threshold:
                candidates.append((overlap, gi, pi))

    candidates.sort(reverse=True)
    matched_g, matched_p = set(), set()
    matches = []
    for overlap, gi, pi in candidates:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        g, p = gold_spans[gi], pred_spans[pi]
        matches.append((gi, pi, g["label"], p["label"], overlap, g["text"], p["text"]))

    unmatched_gold = set(range(len(gold_spans))) - matched_g
    unmatched_pred = set(range(len(pred_spans))) - matched_p
    return matches, unmatched_gold, unmatched_pred

# BERT Matching 

def match_doc_spans_bert(pred_spans: List[Dict], gold_spans: List[Dict], model: SentenceTransformer, threshold: float = 0.8):
    """
    Greedy highest-overlap matching using BERT semantic similarity, label-must-match, thresholded.
    Returns:
      matches: list of tuples (gold_idx, pred_idx, gold_label, pred_label, similarity, gold_text, pred_text)
      unmatched_gold_idx: set of remaining gold indices
      unmatched_pred_idx: set of remaining pred indices
    """
    if not gold_spans or not pred_spans:
        return [], set(range(len(gold_spans))), set(range(len(pred_spans)))
    
    candidates = []
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            if g["label"] != p["label"]:
                continue
            
            # BERT embeddings for both texts
            gold_embedding = model.encode([g["text"]])
            pred_embedding = model.encode([p["text"]])
            
            # Cosine similarity
            similarity = cosine_similarity(gold_embedding, pred_embedding)[0][0]
            
            if similarity >= threshold:
                candidates.append((similarity, gi, pi))

    candidates.sort(reverse=True)
    matched_g, matched_p = set(), set()
    matches = []
    for similarity, gi, pi in candidates:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        g, p = gold_spans[gi], pred_spans[pi]
        matches.append((gi, pi, g["label"], p["label"], similarity, g["text"], p["text"]))

    unmatched_gold = set(range(len(gold_spans))) - matched_g
    unmatched_pred = set(range(len(pred_spans))) - matched_p
    return matches, unmatched_gold, unmatched_pred

# Evaluation  

def evaluate_jaccard(data_jsonl: list, out_dir: str, threshold: float = 0.8):
    y_true, y_pred = [], []
    exact_matches = 0
    partial_matches = 0

    diag_rows = [["doc_id", "type", "gold_label", "pred_label", "overlap", "gold_text", "pred_text"]]

    total_gold_spans = 0
    total_pred_spans = 0

    for i, entry in enumerate(data_jsonl):
        pred_text = entry.get("pred", "").strip()
        gold_text = entry.get("ref", "").strip()

        pred_spans = extract_spans_from_text(pred_text)
        gold_spans = extract_spans_from_text(gold_text)

        total_gold_spans += len(gold_spans)
        total_pred_spans += len(pred_spans)

        # Exact match  
        gold_exact_pool = {(s["label"], normalize_space(s["text"]).lower()) for s in gold_spans}
        pred_exact_pool = {(s["label"], normalize_space(s["text"]).lower()) for s in pred_spans}
        exact_here = len(gold_exact_pool & pred_exact_pool)
        exact_matches += exact_here

        # Greedy label-constrained Jaccard matching
        matches, unmatched_gold, unmatched_pred = match_doc_spans_jaccard(pred_spans, gold_spans, threshold=threshold)
        partial_matches += len(matches)

        # Build classification vectors:
        for gi, pi, gl, pl, ov, gt, pt in matches:
            y_true.append(gl)
            y_pred.append(pl)
            diag_rows.append([i, "MATCH", gl, pl, f"{ov:.3f}", gt, pt])

        for gi in unmatched_gold:
            gl = gold_spans[gi]["label"]
            gt = gold_spans[gi]["text"]
            y_true.append(gl)
            y_pred.append("O")
            diag_rows.append([i, "MISS", gl, "O", "0.000", gt, ""])

        for pi in unmatched_pred:
            pl = pred_spans[pi]["label"]
            pt = pred_spans[pi]["text"]
            y_true.append("O")
            y_pred.append(pl)
            diag_rows.append([i, "SPURIOUS", "O", pl, "0.000", "", pt])

    labels_all = ["Implicit", "Explicit", "O"]
    
    print("\n=== Jaccard Report (threshold ≥ {:.2f}) ===".format(threshold))
    print(classification_report(y_true, y_pred, labels=labels_all, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    # Save reports
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, labels=labels_all, digits=4, zero_division=0))
        f.write("\n\nLabels: " + ", ".join(labels_all) + "\n")
        f.write(np.array2string(cm, separator=", "))

    with open(os.path.join(out_dir, "confusion_matrix.txt"), "w") as f:
        f.write("Labels: " + ", ".join(labels_all) + "\n")
        f.write(np.array2string(cm, separator=", "))

    # Summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Documents compared: {len(data_jsonl)}\n")
        f.write(f"Total gold spans: {total_gold_spans}\n")
        f.write(f"Total pred spans: {total_pred_spans}\n")
        f.write(f"Exact matches (label + normalized text): {exact_matches}\n")
        f.write(f"Partial matches (label + Jaccard≥{threshold}): {partial_matches}\n")

    # Diagnostics CSV
    with open(os.path.join(out_dir, "diagnostics.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(diag_rows)


def evaluate_bert(data_jsonl: list, out_dir: str, threshold: float = 0.8):
    print("Loading BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    y_true, y_pred = [], []
    bert_matches = 0

    # Diagnostics CSV
    diag_rows = [["doc_id", "type", "gold_label", "pred_label", "similarity", "gold_text", "pred_text"]]

    total_gold_spans = 0
    total_pred_spans = 0

    for i, entry in enumerate(data_jsonl):
        pred_text = entry.get("pred", "").strip()
        gold_text = entry.get("ref", "").strip()

        pred_spans = extract_spans_from_text(pred_text)
        gold_spans = extract_spans_from_text(gold_text)

        total_gold_spans += len(gold_spans)
        total_pred_spans += len(pred_spans)

        # BERT semantic similarity matching
        matches, unmatched_gold, unmatched_pred = match_doc_spans_bert(pred_spans, gold_spans, model, threshold=threshold)
        bert_matches += len(matches)

        for gi, pi, gl, pl, sim, gt, pt in matches:
            y_true.append(gl)
            y_pred.append(pl)
            diag_rows.append([i, "MATCH", gl, pl, f"{sim:.3f}", gt, pt])

        for gi in unmatched_gold:
            gl = gold_spans[gi]["label"]
            gt = gold_spans[gi]["text"]
            y_true.append(gl)
            y_pred.append("O")
            diag_rows.append([i, "MISS", gl, "O", "0.000", gt, ""])

        for pi in unmatched_pred:
            pl = pred_spans[pi]["label"]
            pt = pred_spans[pi]["text"]
            y_true.append("O")
            y_pred.append(pl)
            diag_rows.append([i, "SPURIOUS", "O", pl, "0.000", "", pt])

    labels_all = ["Implicit", "Explicit", "O"]
    
    print("\n=== BERT Semantic Similarity Report (similarity ≥ {:.2f}) ===".format(threshold))
    print(classification_report(y_true, y_pred, labels=labels_all, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    # Save BERT reports 
    with open(os.path.join(out_dir, "bert_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, labels=labels_all, digits=4, zero_division=0))
        f.write("\n\nLabels: " + ", ".join(labels_all) + "\n")
        f.write(np.array2string(cm, separator=", "))

    with open(os.path.join(out_dir, "bert_confusion_matrix.txt"), "w") as f:
        f.write("Labels: " + ", ".join(labels_all) + "\n")
        f.write(np.array2string(cm, separator=", "))

    # Summary
    with open(os.path.join(out_dir, "bert_summary.txt"), "w") as f:
        f.write(f"Documents compared: {len(data_jsonl)}\n")
        f.write(f"Total gold spans: {total_gold_spans}\n")
        f.write(f"Total pred spans: {total_pred_spans}\n")
        f.write(f"Partial matches (label + similarity≥{threshold}): {bert_matches}\n")

    # Diagnostics CSV
    with open(os.path.join(out_dir, "bert_diagnostics.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(diag_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Evaluation: Jaccard & BERT")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation reports")
    parser.add_argument("--threshold", type=float, default=0.8, help="Token-overlap threshold (Jaccard) and similarity threshold (BERT)")
    
    args = parser.parse_args()

    # Load data 
    data = load_jsonl(args.predictions)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loaded {len(data)} documents. Running Jaccard evaluation...")
    evaluate_jaccard(data, args.output_dir, args.threshold)

    print(f"\nRunning BERT evaluation...")
    evaluate_bert(data, args.output_dir, args.threshold)

    print(f"\nEvaluation complete. Results saved to {args.output_dir}")
