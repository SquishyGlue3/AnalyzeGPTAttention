"""
Causal Tracing & Ablation Experiment on GPT-2 Large
====================================================
Identifies attention heads that serve as factual knowledge conduits by:
1. Running a baseline forward pass and extracting attention matrices.
2. Identifying heads that attend most from the last token to the subject token.
3. Ablating (zeroing) specific heads via forward hooks and measuring probability drop.
"""

import torch
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ──────────────────────────── Configuration ────────────────────────────

MODEL_NAME = "gpt2-large"
INPUT_CSV = "ex1_data.csv"
OUTPUT_CSV = "results.csv"
ATTENTION_THRESHOLD = 0.25  # for Condition C


# ──────────────────────────── Model Loading ────────────────────────────

def load_model():
    """Load GPT-2 Large model and tokenizer (CPU)."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()
    print("Model loaded.")
    return model, tokenizer


# ──────────────────────── Subject Token Mapping ────────────────────────

def find_subject_token_index(prompt: str, subject: str, tokenizer) -> int:
    """
    Find the index of the LAST token of the subject word(s) within the
    tokenized prompt. Handles sub-word tokenization and multiple occurrences
    by choosing the last occurrence of the subject in the prompt string.

    Returns the token index (0-based position in the tokenized sequence).
    """
    # Find the character-level position of the subject in the prompt (last occurrence)
    char_pos = prompt.rfind(subject)
    if char_pos == -1:
        # Try case-insensitive
        char_pos = prompt.lower().rfind(subject.lower())
    if char_pos == -1:
        raise ValueError(f"Subject '{subject}' not found in prompt '{prompt}'")

    # Character position of the last character of the subject
    subject_end_char = char_pos + len(subject) - 1

    # Tokenize the full prompt
    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)

    # GPT2Tokenizer doesn't natively support return_offsets_mapping via __call__
    # so we build the mapping manually.
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_strings = [tokenizer.decode([t]) for t in tokens]

    # Reconstruct character offsets by matching token strings to the prompt
    offsets = []
    current_pos = 0
    for ts in token_strings:
        # GPT-2 BPE uses Ġ (U+0120) to represent a leading space in tokens
        # The decoded string will have the actual space character
        # Find where this token's text appears starting from current_pos
        decoded = ts  # e.g. " France" or "France"

        # Find the token text in the prompt starting from current_pos
        idx = prompt.find(decoded, current_pos)
        if idx == -1:
            # Try stripping leading space from decoded and matching
            stripped = decoded.lstrip()
            idx = prompt.find(stripped, current_pos)
            if idx == -1:
                # Fallback: just advance by length
                idx = current_pos

        start = idx
        end = start + len(decoded.lstrip() if idx != prompt.find(decoded, current_pos) else decoded)
        offsets.append((start, end))
        current_pos = end

    # Find the token whose span covers the last character of the subject
    last_subject_token_idx = None
    for i, (start, end) in enumerate(offsets):
        if start <= subject_end_char < end:
            last_subject_token_idx = i
            # Don't break - keep going to handle overlaps, take the last match

    if last_subject_token_idx is None:
        # Fallback: find token that best covers the subject end
        # Use a simpler approach: encode prefix up to subject end
        prefix = prompt[: subject_end_char + 1]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        last_subject_token_idx = len(prefix_tokens) - 1

    return last_subject_token_idx


def find_subject_token_index_robust(prompt: str, subject: str, tokenizer) -> int:
    """
    Robust method: encode the prefix of the prompt up to and including the
    subject, and the last token of that encoding is the subject's last token.
    """
    # Find character position of subject (last occurrence)
    char_pos = prompt.rfind(subject)
    if char_pos == -1:
        # case-insensitive fallback
        lower_prompt = prompt.lower()
        lower_subject = subject.lower()
        char_pos = lower_prompt.rfind(lower_subject)
    if char_pos == -1:
        raise ValueError(f"Subject '{subject}' not found in prompt '{prompt}'")

    subject_end_char = char_pos + len(subject)
    prefix = prompt[:subject_end_char]

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    # The last token of the prefix encoding corresponds to the last subject token
    last_subject_token_idx = len(prefix_tokens) - 1
    return last_subject_token_idx


# ─────────────────────── Attention Head Analysis ───────────────────────

def get_top_heads(attentions, last_token_idx: int, subject_token_idx: int, n_top: int = 3):
    """
    From attention matrices, find heads with highest attention weight from
    last_token -> subject_token.

    Args:
        attentions: tuple of (n_layers,) tensors, each (batch, n_heads, seq, seq)
        last_token_idx: index of the last token in the sequence
        subject_token_idx: index of the last token of the subject
        n_top: how many top heads to return

    Returns:
        all_weights: list of (layer, head, weight) for ALL heads, sorted desc
        top_n: the top-n entries
        threshold_heads: heads with weight > ATTENTION_THRESHOLD
    """
    head_weights = []

    for layer_idx, attn in enumerate(attentions):
        # attn shape: (batch, n_heads, seq_len, seq_len)
        # We want attn[0, head, last_token_idx, subject_token_idx]
        n_heads = attn.shape[1]
        for head_idx in range(n_heads):
            w = attn[0, head_idx, last_token_idx, subject_token_idx].item()
            head_weights.append((layer_idx, head_idx, w))

    # Sort by weight descending
    head_weights.sort(key=lambda x: x[2], reverse=True)

    top_n = head_weights[:n_top]
    threshold_heads = [(l, h, w) for l, h, w in head_weights if w > ATTENTION_THRESHOLD]

    return head_weights, top_n, threshold_heads


# ──────────────────────── Ablation via Hooks ───────────────────────────

def make_ablation_hook(head_indices_to_zero, head_dim=64):
    """
    Create a forward hook that zeros out specific attention heads.

    In GPT-2, the attention module (GPT2Attention) forward() returns:
        (attn_output, present_key_value) when output_attentions=False
        (attn_output, present_key_value, attn_weights) when output_attentions=True

    attn_output shape: (batch, seq_len, embed_dim)
    Each head occupies a contiguous slice of size head_dim in the last dimension.

    Args:
        head_indices_to_zero: list of head indices (0-based) to ablate
        head_dim: dimension per head (64 for gpt2-large)
    """
    def hook_fn(module, input, output):
        # output is a tuple: (attn_output, present, [attn_weights])
        attn_output = output[0]  # (batch, seq_len, embed_dim)

        # Clone to avoid in-place modification issues
        modified = attn_output.clone()
        for head_idx in head_indices_to_zero:
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            modified[:, :, start:end] = 0.0

        # Reconstruct output tuple
        return (modified,) + output[1:]

    return hook_fn


def run_with_ablation(model, input_ids, heads_to_ablate, head_dim=64):
    """
    Run forward pass with specified heads ablated.

    Args:
        model: GPT-2 model
        input_ids: tokenized input (1, seq_len)
        heads_to_ablate: list of (layer_idx, head_idx) tuples
        head_dim: dimension per head

    Returns:
        logits: output logits (1, seq_len, vocab_size)
    """
    hooks = []

    # Group heads by layer for efficiency
    layer_heads = {}
    for layer_idx, head_idx in heads_to_ablate:
        layer_heads.setdefault(layer_idx, []).append(head_idx)

    try:
        for layer_idx, head_indices in layer_heads.items():
            hook_fn = make_ablation_hook(head_indices, head_dim)
            handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook_fn)
            hooks.append(handle)

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=False)
            logits = outputs.logits

        return logits
    finally:
        # Always remove hooks
        for h in hooks:
            h.remove()


# ────────────────────────── Probability Extraction ─────────────────────

def get_target_probability(logits, target_token_id: int) -> float:
    """
    Get the probability of the target token from the last position's logits.

    Args:
        logits: (1, seq_len, vocab_size)
        target_token_id: the token id we want the probability for

    Returns:
        probability (float)
    """
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = torch.softmax(last_logits, dim=0)
    return probs[target_token_id].item()


# ────────────────────────── Main Experiment ────────────────────────────

def process_row(model, tokenizer, prompt, subject, target_token, head_dim=64):
    """
    Process a single data row through all experimental conditions.

    Returns dict with baseline_prob, condition deltas, and max head info.
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    seq_len = input_ids.shape[1]
    last_token_idx = seq_len - 1

    # Find subject token index
    subject_token_idx = find_subject_token_index_robust(prompt, subject, tokenizer)

    # Clamp to valid range
    subject_token_idx = min(subject_token_idx, last_token_idx)

    # Tokenize target token - try with and without leading space
    target_candidates = [
        tokenizer.encode(" " + target_token, add_special_tokens=False),
        tokenizer.encode(target_token, add_special_tokens=False),
    ]
    # Use the first token of the encoding (the main token)
    target_token_id = target_candidates[0][0]  # typically with space prefix

    # ── Baseline Pass ──
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        baseline_logits = outputs.logits
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

    baseline_prob = get_target_probability(baseline_logits, target_token_id)

    # ── Identify top heads ──
    all_weights, top3, threshold_heads = get_top_heads(
        attentions, last_token_idx, subject_token_idx, n_top=3
    )

    max_head_layer = top3[0][0]
    max_head_index = top3[0][1]

    # ── Condition A: Zero single top head ──
    heads_a = [(top3[0][0], top3[0][1])]
    logits_a = run_with_ablation(model, input_ids, heads_a, head_dim)
    prob_a = get_target_probability(logits_a, target_token_id)

    # ── Condition B: Zero top 3 heads ──
    heads_b = [(l, h) for l, h, _ in top3]
    logits_b = run_with_ablation(model, input_ids, heads_b, head_dim)
    prob_b = get_target_probability(logits_b, target_token_id)

    # ── Condition C: Zero all heads above threshold ──
    if threshold_heads:
        heads_c = [(l, h) for l, h, _ in threshold_heads]
    else:
        # If no heads above threshold, no ablation (delta = 0)
        heads_c = []

    if heads_c:
        logits_c = run_with_ablation(model, input_ids, heads_c, head_dim)
        prob_c = get_target_probability(logits_c, target_token_id)
    else:
        prob_c = baseline_prob

    # ── Compute relative deltas ──
    if baseline_prob > 0:
        delta_a = (baseline_prob - prob_a) / baseline_prob
        delta_b = (baseline_prob - prob_b) / baseline_prob
        delta_c = (baseline_prob - prob_c) / baseline_prob
    else:
        delta_a = delta_b = delta_c = 0.0

    return {
        "baseline_prob": baseline_prob,
        "condition_a_delta": delta_a,
        "condition_b_delta": delta_b,
        "condition_c_delta": delta_c,
        "max_head_layer": max_head_layer,
        "max_head_index": max_head_index,
    }


def main():
    # Load model
    model, tokenizer = load_model()

    # Determine head dimension
    n_heads = model.config.n_head      # 20 for gpt2-large
    n_embd = model.config.n_embd       # 1280 for gpt2-large
    head_dim = n_embd // n_heads       # 64

    print(f"Model config: {model.config.n_layer} layers, {n_heads} heads, "
          f"head_dim={head_dim}, embed_dim={n_embd}")

    # Load dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} prompts from {INPUT_CSV}")

    results = []

    for idx, row in df.iterrows():
        prompt = str(row["Prompt"]).strip()
        subject = str(row["Subject Word(s)"]).strip()
        target = str(row["Target Token"]).strip()
        domain = str(row.get("Domain", "")).strip()

        print(f"\n[{idx+1}/{len(df)}] Prompt: \"{prompt}\"")
        print(f"  Subject: \"{subject}\" | Target: \"{target}\" | Domain: {domain}")

        try:
            result = process_row(model, tokenizer, prompt, subject, target, head_dim)
            result["prompt_id"] = idx + 1

            print(f"  Baseline prob: {result['baseline_prob']:.6f}")
            print(f"  Cond A delta:  {result['condition_a_delta']:.4f}")
            print(f"  Cond B delta:  {result['condition_b_delta']:.4f}")
            print(f"  Cond C delta:  {result['condition_c_delta']:.4f}")
            print(f"  Max head: layer={result['max_head_layer']}, "
                  f"head={result['max_head_index']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "prompt_id": idx + 1,
                "baseline_prob": 0.0,
                "condition_a_delta": 0.0,
                "condition_b_delta": 0.0,
                "condition_c_delta": 0.0,
                "max_head_layer": -1,
                "max_head_index": -1,
            }

        results.append(result)

    # Build output DataFrame with exact column order
    out_df = pd.DataFrame(results)[
        [
            "prompt_id",
            "baseline_prob",
            "condition_a_delta",
            "condition_b_delta",
            "condition_c_delta",
            "max_head_layer",
            "max_head_index",
        ]
    ]

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
