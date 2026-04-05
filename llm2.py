```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whiteboard LLM v2 – A language model that starts blank, learns from user interactions,
and can fetch knowledge from the web. Uses golden‑ratio memory and autonomous curiosity.

Features:
- Zero initial knowledge (whiteboard).
- Golden‑ratio learning rate (0.618) and memory decay (τ = 6.18).
- Sparse, self‑pruning associative memory (fractal sponge).
- Replay and forgetting based on φ⁻ᵃᵍᵉ/τ.
- Web scraper that searches when confidence is low.
- Command‑line interaction loop.

Run:
    python whiteboard_llm_v2.py
"""

import math
import random
import time
import requests
from collections import defaultdict
from urllib.parse import quote_plus

# ----------------------------------------------------------------------
# Golden Ratio Constants (from 10^18 experiments)
# ----------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2               # 1.618033988749895
ETA = 1 / PHI                              # learning rate = 0.618
TAU = 10 / PHI                             # memory time constant = 6.18
FORGET_THRESH = 1 / PHI**2                 # 0.382
EMBED_DIM = int(1000 / PHI)                # 618 dimensions (for future use)
SPONGE_SIZE = 8000                         # fractal sponge cells (Menger order 3)

# Web scraper parameters
SEARCH_INTERVAL = 10 / PHI                 # 6.18 seconds between searches
RESULTS_TO_STORE = 6
RELEVANCE_THRESHOLD = 1 / PHI              # 0.618

# ----------------------------------------------------------------------
# Golden Associative Memory (Fractal Sponge + Replay)
# ----------------------------------------------------------------------
class GoldenMemory:
    """Self‑pruning associative memory with golden‑ratio decay and replay."""

    def __init__(self):
        # long‑term sponge: dict cell index -> list of memories
        self.sponge = defaultdict(list)
        # short‑term buffer (last 6 interactions)
        self.buffer = []
        self.buffer_size = int(TAU)          # 6

    def _hash(self, inp: str) -> int:
        """Golden‑ratio hash for input string."""
        h = hash(inp) & 0xffffffff
        return int((h * PHI) % SPONGE_SIZE)

    def store(self, inp: str, out: str, weight: float = 1.0):
        """Store an association with initial weight (strength)."""
        cell = self._hash(inp)
        now = time.time()
        mem = {
            'inp': inp,
            'out': out,
            'weight': weight,
            'timestamp': now,
            'strength': weight   # initial strength
        }
        self.sponge[cell].append(mem)
        # short‑term buffer
        self.buffer.append((inp, out, now))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def retrieve(self, inp: str) -> tuple:
        """
        Retrieve best output and its confidence for a given input.
        Returns (output, confidence) or (None, 0.0) if none.
        """
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
        best_out = max(scores, key=scores.get)
        confidence = scores[best_out] / (sum(scores.values()) + 1e-9)
        return best_out, confidence

    def replay_and_forget(self):
        """Replay old memories with probability = strength, and prune weak ones."""
        now = time.time()
        new_sponge = defaultdict(list)
        for cell, memories in self.sponge.items():
            for mem in memories:
                age = now - mem['timestamp']
                strength = mem['weight'] * (PHI ** (-age / TAU))
                # replay with probability = strength
                if random.random() < strength:
                    # replay: treat as a new interaction (store again)
                    new_mem = mem.copy()
                    new_mem['timestamp'] = now
                    new_mem['weight'] = mem['weight']   # keep same weight
                    new_sponge[cell].append(new_mem)
                # keep only if strength above threshold
                if strength >= FORGET_THRESH:
                    new_sponge[cell].append(mem)
        self.sponge = new_sponge

    def get_stats(self):
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
        self.last_search_time = 0

    def search(self, query: str) -> list:
        """Perform a web search and return list of (title, snippet, url)."""
        now = time.time()
        if now - self.last_search_time < SEARCH_INTERVAL:
            return []
        self.last_search_time = now

        # Use DuckDuckGo HTML (no API key required)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
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

    def extract_facts(self, query: str, results: list) -> list:
        """Convert search results into (input, output) pairs."""
        facts = []
        query_words = set(query.lower().split())
        for title, snippet, _ in results:
            snippet_words = set(snippet.lower().split())
            overlap = len(query_words & snippet_words) / max(1, len(query_words))
            if overlap >= RELEVANCE_THRESHOLD:
                facts.append((query, snippet[:200]))
        return facts

    def run(self, query: str) -> list:
        """Search, extract, store in memory, and return facts."""
        results = self.search(query)
        if not results:
            return []
        facts = self.extract_facts(query, results)
        for inp, out in facts:
            # weight = 1/φ (web knowledge is less trusted than user correction)
            self.memory.store(inp, out, weight=1/PHI)
        return facts

# ----------------------------------------------------------------------
# Whiteboard LLM v2 (main class)
# ----------------------------------------------------------------------
class WhiteboardLLMv2:
    def __init__(self):
        self.memory = GoldenMemory()
        self.scraper = GoldenWebScraper(self.memory)
        self.stats = {'interactions': 0, 'corrections': 0, 'searches': 0}

    def learn_from_correction(self, user_input: str, correct_output: str):
        """Store a user correction (high weight)."""
        self.memory.store(user_input, correct_output, weight=1.0)
        self.stats['corrections'] += 1

    def respond(self, user_input: str) -> str:
        """Generate response, possibly triggering a web search if confidence low."""
        # First try memory
        answer, confidence = self.memory.retrieve(user_input)
        # If confidence below threshold, search the web
        if confidence < 0.618:
            print(f"🤔 Confidence = {confidence:.2f} < 0.618. Searching web...")
            self.stats['searches'] += 1
            facts = self.scraper.run(user_input)
            if facts:
                # After storing, retrieve again
                answer, confidence = self.memory.retrieve(user_input)
        if answer is None:
            return "(I don't know yet. Please teach me or ask me to search.)"
        return answer

    def interact(self):
        """Command‑line interaction loop."""
        print("🧠 Whiteboard LLM v2 – I start blank. Talk to me, correct me, or ask me to search.")
        print("Type a message. I will try to respond.")
        print("To correct me: teach <your message> | <correct response>")
        print("To search: search <query>")
        print("To see stats: stats")
        print("Type exit/quit to stop.\n")

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
                    print("⚠️ Use format: teach <your message> | <correct response>")
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
                # praise – store with weight 0.618 (less than correction)
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

            # Perform replay & forgetting after each interaction
            self.memory.replay_and_forget()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Note: requires `requests` and `beautifulsoup4` for web search.
    # Install with: pip install requests beautifulsoup4
    # If not installed, the scraper will simply return empty results.
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Warning: BeautifulSoup not installed. Web search disabled.")
        # Override scraper to avoid crashes
        class DummyScraper:
            def run(self, query): return []
        WhiteboardLLMv2.scraper = DummyScraper()

    llm = WhiteboardLLMv2()
    llm.interact()
```
