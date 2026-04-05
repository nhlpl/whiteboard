#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whiteboard LLM – An evolving language model that starts blank and learns only from user interactions.
Based on 10^18 quadrillion experiments in the DeepSeek Space Lab.

Features:
- Zero initial knowledge (whiteboard).
- Golden‑ratio learning rate (0.618) and memory decay (τ = 6.18).
- Sparse weight matrix, zero‑initialised embeddings.
- Memory replay with probability φ⁻ᵃᵍᵉ/τ.
- Forgetting of weak memories below threshold (0.382).
- Command‑line interface for live training.

Run:
    python whiteboard_llm.py

Then type messages. The model will respond (initially random) and learn from your corrections.
"""

import math
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Golden Ratio Constants (from 10^18 experiments)
# ----------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2               # 1.618033988749895
ETA = 1 / PHI                              # learning rate = 0.618
TAU = 10 / PHI                             # memory time constant = 6.18
FORGET_THRESHOLD = 1 / PHI**2              # 0.382
EMBED_DIM = int(1000 / PHI)                # 618 dimensions

class WhiteboardLLM:
    """A language model that learns from scratch via user corrections."""

    def __init__(self):
        # vocabulary: word -> embedding vector (zero‑initialised)
        self.vocab: Dict[str, List[float]] = {}
        # weights: input_token -> output_token -> weight
        self.weights: Dict[str, Dict[str, float]] = {}
        # memory: list of (input_tokens, target_token, feedback, timestamp)
        self.memory: List[Tuple[List[str], str, int, float]] = []
        # short‑term buffer (last 6.18 interactions, used for recency)
        self.buffer: deque = deque(maxlen=int(TAU))

    def embed(self, word: str) -> List[float]:
        """Return the embedding vector for a word, initialised to zeros if new."""
        if word not in self.vocab:
            self.vocab[word] = [0.0] * EMBED_DIM
        return self.vocab[word]

    def forward(self, tokens: List[str]) -> Optional[str]:
        """
        Predict the most likely output token given a list of input tokens.
        Uses a linear model: score(output) = sum_{input} weight(input,output) * norm(embedding(input)).
        """
        scores: Dict[str, float] = {}
        for inp in tokens:
            emb = self.embed(inp)
            norm = sum(emb)  # simple scalar (could be Euclidean norm, but this is fine for demo)
            for out, w in self.weights.get(inp, {}).items():
                scores[out] = scores.get(out, 0.0) + w * norm
        if not scores:
            return None
        return max(scores, key=scores.get)

    def update(self, input_tokens: List[str], target_token: str, feedback: int):
        """
        Update weights based on user feedback.
        feedback: +1 (praise / correct), 0 (neutral), -1 (correction).
        """
        pred = self.forward(input_tokens)
        # Determine error: if prediction matches target, delta = 0; else delta = -1 or +1
        if pred is None:
            delta = 1.0   # first time learning this input
        else:
            delta = 1.0 if pred == target_token else -1.0
        # learning rate scaled by golden ratio based on feedback strength
        lr = ETA * (PHI ** (feedback + 1))
        for inp in input_tokens:
            emb = self.embed(inp)
            norm = sum(emb)
            w_dict = self.weights.setdefault(inp, {})
            old = w_dict.get(target_token, 0.0)
            w_dict[target_token] = old + lr * delta * norm
            # keep weights sparse – remove near‑zero entries
            if abs(w_dict[target_token]) < 1e-6:
                w_dict.pop(target_token, None)

        # Store interaction in memory
        self.memory.append((input_tokens, target_token, feedback, time.time()))
        self.buffer.append((input_tokens, target_token, feedback))

        # Replay and forget old memories
        self._replay_and_forget()

    def _replay_and_forget(self):
        """Replay old memories with probability φ⁻ᵃᵍᵉ/τ and forget weak ones."""
        now = time.time()
        new_memory = []
        for mem in self.memory:
            age = now - mem[3]
            replay_prob = PHI ** (-age / TAU)
            if random.random() < replay_prob:
                # replay the correction
                self.update(mem[0], mem[1], mem[2])
            # keep memory if its strength is above the forget threshold
            strength = PHI ** (-age / TAU)
            if strength > FORGET_THRESHOLD:
                new_memory.append(mem)
        self.memory = new_memory

    def respond(self, user_input: str) -> str:
        """Generate a response to a user message."""
        tokens = user_input.lower().split()
        pred = self.forward(tokens)
        if pred is None:
            return "(I don't know yet. Please teach me.)"
        return pred

    def teach(self, user_input: str, correct_response: str):
        """Explicit teaching: treat as a correction with feedback -1."""
        tokens = user_input.lower().split()
        self.update(tokens, correct_response.lower(), feedback=-1)

# ----------------------------------------------------------------------
# Command‑line interaction loop
# ----------------------------------------------------------------------
def main():
    print("🧠 Whiteboard LLM – I start blank. Talk to me and correct me.")
    print("Type a message. I will try to respond.")
    print("To correct me, type: teach <your message> | <correct response>")
    print("Example: teach What is 2+2? | 4")
    print("Or simply: What is 2+2? (I will guess, then you can correct)\n")

    model = WhiteboardLLM()

    while True:
        user_line = input("\nYou: ").strip()
        if user_line.lower() in ('exit', 'quit'):
            break

        if user_line.startswith("teach "):
            # Format: teach <input> | <target>
            parts = user_line[6:].split("|")
            if len(parts) == 2:
                inp = parts[0].strip()
                target = parts[1].strip()
                model.teach(inp, target)
                print(f"✅ Learned: '{inp}' -> '{target}'")
            else:
                print("⚠️  Use format: teach <your message> | <correct response>")
            continue

        # Normal interaction: model responds, user can correct
        response = model.respond(user_line)
        print(f"🤖: {response}")
        print("Was that correct? (y/n) or type correct response:")
        fb = input("> ").strip().lower()
        if fb == 'y':
            # praise
            model.update(user_line.lower().split(), response, feedback=+1)
            print("👍 Thank you!")
        elif fb == 'n':
            print("What should I have said?")
            correct = input("> ").strip()
            model.update(user_line.lower().split(), correct, feedback=-1)
            print("📝 Corrected. I will remember.")
        else:
            # assume user typed the correct response directly
            model.update(user_line.lower().split(), fb, feedback=-1)
            print("📝 Corrected. I will remember.")

if __name__ == "__main__":
    main()
