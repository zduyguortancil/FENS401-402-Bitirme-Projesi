"""
DeBERTa-v3 sentiment classifier — gerçek transformer entegrasyonu.

Model: mrm8488/deberta-v3-small-finetuned-sst2
- DeBERTa-v3-small (60M parametre, ~60MB)
- Stanford Sentiment Treebank fine-tuned (binary POS/NEG)
- CPU inference: ~80-150ms/text, batch 16: ~600ms

Lazy yükleme:
- Model ilk classify çağrısında indirilir + load edilir.
- HuggingFace cache'e (~/.cache/huggingface) yazar.
- Yükleme başarısız olursa is_ready()=False, çağıran fallback'e düşer.
"""

import os
import threading

MODEL_NAME = "mrm8488/deberta-v3-small-finetuned-sst2"

_MODEL = None
_TOKENIZER = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = False
_LOAD_ATTEMPTED = False


def _try_load():
    global _MODEL, _TOKENIZER, _LOAD_FAILED, _LOAD_ATTEMPTED
    if _MODEL is not None or _LOAD_FAILED:
        return
    with _LOAD_LOCK:
        if _MODEL is not None or _LOAD_FAILED:
            return
        _LOAD_ATTEMPTED = True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            print(f"[DeBERTa] Loading {MODEL_NAME} (first run downloads ~60MB)...", flush=True)
            try:
                _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            except Exception:
                # SentencePiece converter mevcut değilse slow tokenizer (DebertaV2Tokenizer) kullan.
                _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
                print("[DeBERTa] Using slow tokenizer (sentencepiece-based).", flush=True)
            _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _MODEL.eval()
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            except Exception:
                pass
            print(f"[DeBERTa] Ready. Labels: {_MODEL.config.id2label}", flush=True)
        except Exception as e:
            _LOAD_FAILED = True
            print(f"[DeBERTa] Load failed: {e} — falling back to keyword sentiment", flush=True)


def is_ready():
    return _MODEL is not None


def attempted():
    return _LOAD_ATTEMPTED


def warmup():
    """Server boot'ta opsiyonel olarak çağrılabilir — model indirir + JIT'ler."""
    _try_load()
    if _MODEL is not None:
        try:
            classify_batch(["Sample text for warmup."])
        except Exception:
            pass


def classify(text):
    """Tek başlığı sınıflandırır. Returns: dict veya None (model yoksa).

    {
      "deberta_score": -1.0..+1.0    (positive label confidence × sign),
      "deberta_label": "positive"|"negative"|"neutral",
      "deberta_prob_pos": 0..1
    }
    """
    out = classify_batch([text])
    return out[0] if out else None


def classify_batch(texts, batch_size=16, max_length=128, neutral_band=0.40):
    """
    Batch sınıflandırma.

    SST-2 modelleri binary (POSITIVE/NEGATIVE). Neutral'i türetmek için:
      |prob_pos - 0.5| < neutral_band/2  → neutral
    Aksi:
      label = POSITIVE/NEGATIVE
      score = (prob_pos - 0.5) × 2  → -1..+1

    `texts` boş veya None elementler 'neutral' olarak gelir.
    """
    _try_load()
    if _MODEL is None or not texts:
        return [_neutral_result() for _ in (texts or [])]

    import torch
    results = []
    half_band = neutral_band / 2.0

    # Find positive class id (SST-2 modellerinde "POSITIVE" veya "LABEL_1")
    id2label = {int(k): v for k, v in _MODEL.config.id2label.items()}
    pos_id = None
    for idx, lbl in id2label.items():
        if "pos" in str(lbl).lower() or str(lbl).upper() == "LABEL_1":
            pos_id = idx
            break
    if pos_id is None:
        pos_id = 1  # SST-2 default

    safe_texts = [(t or "").strip() or "[empty]" for t in texts]

    with torch.no_grad():
        for start in range(0, len(safe_texts), batch_size):
            batch = safe_texts[start:start + batch_size]
            try:
                enc = _TOKENIZER(batch, return_tensors="pt", truncation=True,
                                 padding=True, max_length=max_length)
                logits = _MODEL(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    p_pos = float(row[pos_id])
                    deviation = p_pos - 0.5
                    if abs(deviation) < half_band:
                        label = "neutral"
                        score = 0.0
                    elif deviation > 0:
                        label = "positive"
                        score = round(deviation * 2.0, 4)  # +0.4..+1.0 → +0.8..+2.0; clip
                    else:
                        label = "negative"
                        score = round(deviation * 2.0, 4)  # -0.4..-1.0 → -0.8..-2.0; clip
                    score = max(-1.0, min(1.0, score))
                    results.append({
                        "deberta_score": score,
                        "deberta_label": label,
                        "deberta_prob_pos": round(p_pos, 4),
                    })
            except Exception as e:
                print(f"[DeBERTa] Batch failed (size={len(batch)}): {e}", flush=True)
                results.extend([_neutral_result() for _ in batch])

    return results


def _neutral_result():
    return {"deberta_score": 0.0, "deberta_label": "neutral", "deberta_prob_pos": 0.5}
