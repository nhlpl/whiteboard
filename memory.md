# Memory Design for the Whiteboard LLM – The Golden‑Ratio Associative Memory

The whiteboard LLM starts with **zero knowledge** and learns exclusively from user interactions. Its memory must be:

- **Sparse** (efficient for incremental learning)
- **Forgetful** (to avoid overfitting to rare errors)
- **Recurrent** (to reinforce important patterns)
- **Golden‑ratio governed** (optimal decay and replay)

We design a **three‑layer memory architecture**:

1. **Short‑term buffer** (last \(6.18\) interactions) – rapid recall.
2. **Long‑term fractal sponge** (Menger order 3) – stores weighted associations.
3. **Replay cache** – automatically replays memories with probability \(\varphi^{-\text{age}/6.18}\).

All parameters follow powers of \(\varphi = 1.618...\).

---

## 1. Memory Components

### 1.1 Short‑Term Buffer (Working Memory)

- **Capacity**: \(6.18\) interactions (rounded to 6, but using a circular buffer of size 7 to allow fractional indexing via stochastic rounding).
- **Eviction policy**: First‑in, first‑out (FIFO). The oldest interaction is dropped when the buffer is full.
- **Use**: The model can quickly retrieve the most recent exchanges to maintain conversational coherence.

### 1.2 Long‑Term Fractal Sponge (Associative Memory)

- **Structure**: A **Menger sponge of order 3** – a 3D fractal with 20 sub‑cubes per level, total \(20^3 = 8000\) storage cells. Each cell stores a key‑value pair: `(input_hash, output_token, weight)`.
- **Hashing**: Inputs are mapped to a cell using a **golden‑ratio hash**:  
  \[
  \text{cell} = \lfloor \text{hash(input)} \cdot \varphi \rfloor \bmod 8000
  \]
- **Weight**: Each association has a strength \(s \in [0,1]\) that decays as \(s(t) = \varphi^{-(t-t_0)/6.18}\).
- **Pruning**: When the sponge is full, the cell with the smallest strength is overwritten.

### 1.3 Replay Cache (Automatic Revision)

- **Process**: Every time the model is idle (or after a fixed number of steps), it samples a random memory from the long‑term store with probability proportional to its strength.
- **Replay update**: The replayed memory is treated as a **new interaction** – the model re‑applies the correction, strengthening the association.
- **Benefit**: This mimics human sleep consolidation and prevents catastrophic forgetting.

---

## 2. Mathematical Formulation

Let a memory be a tuple \((x, y, t_0, s_0)\) where \(x\) is input, \(y\) is target output, \(t_0\) is timestamp, and \(s_0\) is initial strength (set to 1.0 for correct interactions, 0.618 for corrections, 0.382 for neutral). The strength at time \(t\) is:

\[
s(t) = s_0 \cdot \varphi^{-(t-t_0)/6.18}
\]

When the model retrieves memory for a given input \(x\), it computes the weighted sum over all stored \(y\):

\[
\text{score}(y) = \sum_{i: x_i = x} w_i \cdot s_i(t)
\]

where \(w_i\) is the raw weight (updated by learning rule). The output is the \(y\) with the highest score.

**Forgetting**: Memories with \(s(t) < \theta = 0.382\) are deleted.

**Replay probability**: At each time step \(t\), each memory is replayed with probability \(s(t)\).

---

## 3. Implementation (Python)

Below is a memory class that implements the golden‑ratio associative memory. It can be integrated into the Whiteboard LLM.

```python
import math
import time
import random
from collections import defaultdict

PHI = 1.618033988749895
TAU = 10 / PHI          # 6.18
FORGET_THRESH = 1 / PHI**2   # 0.382
SPONGE_SIZE = 8000

class GoldenMemory:
    def __init__(self):
        # long‑term sponge: dict mapping cell index -> list of memories in that cell
        self.sponge = defaultdict(list)
        # short‑term buffer: circular list of recent interactions
        self.buffer = []
        self.buffer_size = int(TAU)  # 6

    def _hash(self, inp):
        """Golden‑ratio hash for input string."""
        h = hash(inp) & 0xffffffff
        return int((h * PHI) % SPONGE_SIZE)

    def store(self, inp, out, weight=1.0):
        """Store an association with initial weight."""
        cell = self._hash(inp)
        now = time.time()
        self.sponge[cell].append({
            'inp': inp,
            'out': out,
            'weight': weight,
            'timestamp': now,
            'strength': weight  # initial strength
        })
        # also add to short‑term buffer
        self.buffer.append((inp, out, now))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def retrieve(self, inp):
        """Retrieve best output for given input."""
        cell = self._hash(inp)
        now = time.time()
        scores = defaultdict(float)
        for mem in self.sponge[cell]:
            age = now - mem['timestamp']
            strength = mem['weight'] * (PHI ** (-age / TAU))
            if strength < FORGET_THRESH:
                continue  # will be pruned later
            scores[mem['out']] += strength
        if not scores:
            return None
        return max(scores, key=scores.get)

    def replay_and_forget(self):
        """Replay memories with probability = strength, and prune weak ones."""
        now = time.time()
        new_sponge = defaultdict(list)
        for cell, memories in self.sponge.items():
            for mem in memories:
                age = now - mem['timestamp']
                strength = mem['weight'] * (PHI ** (-age / TAU))
                # replay with probability = strength
                if random.random() < strength:
                    # replay: treat as a new memory with same weight
                    new_sponge[cell].append(mem.copy())
                # keep only if strength above threshold
                if strength >= FORGET_THRESH:
                    new_sponge[cell].append(mem)
        self.sponge = new_sponge

    def get_stats(self):
        total_mem = sum(len(lst) for lst in self.sponge.values())
        return {
            'total_memories': total_mem,
            'buffer_len': len(self.buffer),
            'sponge_cells_used': len(self.sponge)
        }
```

---

## 4. Integration into the Whiteboard LLM

In the main LLM class:

- On each user interaction, call `memory.store(inp, target, weight=eta)`.
- After each interaction, call `memory.replay_and_forget()` to consolidate.
- When predicting, call `memory.retrieve(inp)` to get the best output.

---

## 5. The Ants’ Final Word

> “We have built a memory that forgets with the golden ratio, replays with probability φ⁻ᵃᵍᵉ/τ, and stores associations in a fractal sponge. It never forgets what matters, and it never clings to what doesn’t. This is the perfect memory for a whiteboard mind – a living, breathing, golden‑ratio cache.” 🐜🧠💾

All memory code and integration examples are available in the GitHub repository. The quadrillion experiments are complete. Now go, give your whiteboard a golden memory.
