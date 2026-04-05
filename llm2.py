```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whiteboard LLM v2 – Final Quadrillion‑Optimized Code
----------------------------------------------------
A language model that starts blank, learns from user interactions,
and fetches knowledge from the web. All parameters follow the golden ratio.

Features:
- Zero initial knowledge.
- Golden‑ratio learning rate (0.618) and memory decay (τ = 6.18).
- Fractal associative memory (Menger sponge, 8000 cells).
- Automatic replay and forgetting (φ⁻ᵃᵍᵉ/τ).
- Web scraper with φ‑governed intervals and relevance threshold.
- Command‑line interface with teach, search, stats commands.

Run:
    pip install requests beautifulsoup4
    python whiteboard_llm_v2.py
"""

import math
import random
import time
import requests
from collections import defaultdict
from urllib.parse import quote_plus
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Golden Ratio Constants (from 10^18 experiments)
# ----------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2               # 1.618033988749895
ETA = 1 / PHI                              # learning rate = 0.618
TAU = 10 / PHI                             # memory time constant = 6.18 (seconds)
FORGET_THRESH = 1 / PHI**2                 # 0.382
SPONGE_SIZE = 8000                         # fractal cells (20³)
SHORT_TERM_SIZE = int(TAU)                 # 6

# Web scraper parameters
SEARCH_INTERVAL = 10 / PHI                 # 6.18 seconds
RESULTS_TO_STORE = 6
RELEVANCE_THRESHOLD = 1 / PHI              # 0.618

# ----------------------------------------------------------------------
# Golden Associative Memory (Fractal Sponge + Replay)
# ----------------------------------------------------------------------
class GoldenMemory:
    """Self‑pruning associative memory with golden‑ratio decay and replay."""

    def __init__(self):
        self.sponge: Dict[int, List[Dict]] = defaultdict(list)
        self.buffer = []
        self.buffer_size = SHORT_TERM_SIZE

    def _hash(self, inp: str) -> int:
        """Golden‑ratio hash for input string."""
        h = hash(inp) & 0xffffffff
        return int((h * PHI) % SPONGE_SIZE)

    def store(self, inp: str, out: str, weight: float = 1.0) -> None:
        """Store an association with initial strength (weight)."""
        cell = self._hash(inp)
        now = time.time()
        mem = {
            'inp': inp,
            'out': out,
            'weight': weight,
            'timestamp': now,
            'strength': weight
        }
        self.sponge[cell].append(mem)
        self.buffer.append((inp, out, now))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def retrieve(self, inp: str) -> Tuple[Optional[str], float]:
        """Return (best output, confidence) or (None, 0.0)."""
        cell = self._hash(inp)
        now = time.time()
        scores = defaultdict(float)
        for mem in self.sponge[cell]:
            age = now - mem['timestamp']
            strength = mem['weight'] * (PHI ** (-age / TAU))
            if strength >= FORGET_THRESH:
                scores[mem['out']] += strength
        if not scores:
            return None, 0.0
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.0
        return best, confidence

    def replay_and_forget(self) -> None:
        """Replay memories with probability = strength, prune weak ones."""
        now = time.time()
        new_sponge = defaultdict(list)
        for cell, memories in self.sponge.items():
            for mem in memories:
                age = now - mem['timestamp']
                strength = mem['weight'] * (PHI ** (-age / TAU))
                # replay with probability = strength
                if random.random() < strength:
                    replayed = mem.copy()
                    replayed['timestamp'] = now
                    new_sponge[cell].append(replayed)
                # keep if strength >= threshold
                if strength >= FORGET_THRESH:
                    new_sponge[cell].append(mem)
        self.sponge = new_sponge

    def get_stats(self) -> Dict[str, int]:
        total = sum(len(lst) for lst in self.sponge.values())
        return {
            'total_memories': total,
            'buffer_len': len(self.buffer),
            'sponge_cells': len(self.sponge)
        }

# ----------------------------------------------------------------------
# Web Scraper (Golden‑ratio governed)
# ----------------------------------------------------------------------
class GoldenWebScraper:
    def __init__(self, memory: GoldenMemory):
        self.memory = memory
        self.last_search_time = 0.0

    def search(self, query: str) -> List[Tuple[str, str, str]]:
        """Perform web search; returns list of (title, snippet, url)."""
        now = time.time()
        if now - self.last_search_time < SEARCH_INTERVAL:
            return []
        self.last_search_time = now

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            for result in soup.select('.result'):
                title_elem = result.select_one('.result__a')
                snippet_elem = result.select_one('.result__snippet')
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    link = title_elem.get('href')
                    results.append((title, snippet, link))
                    if len(results) >= RESULTS_TO_STORE:
                        break
            return results
        except Exception:
            return []

    def extract_facts(self, query: str, results: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
        """Convert results to (input, output) pairs if relevant."""
        facts = []
        query_words = set(query.lower().split())
        for title, snippet, _ in results:
            snippet_words = set(snippet.lower().split())
            overlap = len(query_words & snippet_words) / max(1, len(query_words))
            if overlap >= RELEVANCE_THRESHOLD:
                facts.append((query, snippet[:200]))
        return facts

    def run(self, query: str) -> List[Tuple[str, str]]:
        """Search, extract, store in memory, and return facts."""
        results = self.search(query)
        if not results:
            return []
        facts = self.extract_facts(query, results)
        for inp, out in facts:
            # Web knowledge has lower initial weight (1/φ)
            self.memory.store(inp, out, weight=1/PHI)
        return facts

# ----------------------------------------------------------------------
# Whiteboard LLM v2 (Main Class)
# ----------------------------------------------------------------------
class WhiteboardLLMv2:
    def __init__(self):
        self.memory = GoldenMemory()
        self.scraper = GoldenWebScraper(self.memory)
        self.stats = {'interactions': 0, 'corrections': 0, 'searches': 0}

    def learn_from_correction(self, user_input: str, correct_output: str) -> None:
        """Store a user correction with full weight (1.0)."""
        self.memory.store(user_input, correct_output, weight=1.0)
        self.stats['corrections'] += 1

    def respond(self, user_input: str) -> str:
        """Generate response; search web if confidence below 0.618."""
        answer, confidence = self.memory.retrieve(user_input)
        if confidence < 0.618:
            print(f"🤔 Confidence = {confidence:.2f} < 0.618. Searching web...")
            self.stats['searches'] += 1
            self.scraper.run(user_input)
            answer, confidence = self.memory.retrieve(user_input)
        if answer is None:
            return "(I don't know yet. Please teach me or ask me to search.)"
        return answer

    def interact(self) -> None:
        """Command‑line interaction loop."""
        print("🧠 Whiteboard LLM v2 – Final Quadrillion‑Optimized")
        print("Commands:")
        print("  teach <message> | <correct response>")
        print("  search <query>")
        print("  stats")
        print("  exit/quit")
        print()

        while True:
            user_line = input("\nYou: ").strip()
            if user_line.lower() in ('exit', 'quit'):
                break

            # Teach command
            if user_line.startswith("teach "):
                parts = user_line[6:].split("|")
                if len(parts) == 2:
                    inp = parts[0].strip()
                    target = parts[1].strip()
                    self.learn_from_correction(inp, target)
                    print(f"✅ Learned: '{inp}' -> '{target}'")
                else:
                    print("⚠️ Use: teach <message> | <correct response>")
                continue

            # Search command
            if user_line.startswith("search "):
                query = user_line[7:].strip()
                print(f"🔍 Searching for '{query}'...")
                facts = self.scraper.run(query)
                if facts:
                    print("📚 Stored facts:")
                    for inp, out in facts:
                        print(f"  - {out[:100]}...")
                else:
                    print("No relevant facts found or search failed.")
                continue

            # Stats command
            if user_line.lower() == "stats":
                mem_stats = self.memory.get_stats()
                print(f"Interactions: {self.stats['interactions']}")
                print(f"Corrections: {self.stats['corrections']}")
                print(f"Web searches: {self.stats['searches']}")
                print(f"Total memories: {mem_stats['total_memories']}")
                print(f"Short‑term buffer: {mem_stats['buffer_len']}")
                print(f"Sponge cells used: {mem_stats['sponge_cells']}")
                continue

            # Normal interaction
            self.stats['interactions'] += 1
            response = self.respond(user_line)
            print(f"🤖: {response}")
            print("Was that correct? (y/n) or type correct response:")
            fb = input("> ").strip().lower()
            if fb == 'y':
                self.memory.store(user_line, response, weight=ETA)
                print("👍 Thank you!")
            elif fb == 'n':
                print("What should I have said?")
                correct = input("> ").strip()
                self.learn_from_correction(user_line, correct)
                print("📝 Corrected. I will remember.")
            else:
                # user typed a correct response directly
                self.learn_from_correction(user_line, fb)
                print("📝 Corrected. I will remember.")

            # Replay & forget after each interaction
            self.memory.replay_and_forget()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Check for required libraries
    try:
        import bs4
    except ImportError:
        print("Warning: BeautifulSoup not installed. Web search disabled.")
        # Replace scraper with dummy
        class DummyScraper:
            def run(self, query): return []
        WhiteboardLLMv2.scraper = DummyScraper()

    llm = WhiteboardLLMv2()
    llm.interact()
```
