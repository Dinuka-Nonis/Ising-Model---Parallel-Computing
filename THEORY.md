# 🧲 Understanding the Ising Model — From Zero
### SE3082 Parallel Computing | IT23614130
### Everything you need to understand the theory behind this assignment

---

## Part 1: What Even Is This?

Imagine you have a giant grid of tiny magnets. Each magnet can only point in one of two directions: **UP (↑)** or **DOWN (↓)**. That's it. Nothing else.

In code, we represent:
- **UP** = `+1`
- **DOWN** = `-1`

This is the **Ising Model**. It was invented in 1920 by physicist Wilhelm Lenz and solved by his student Ernst Ising in 1925.

Despite being incredibly simple (just a grid of +1s and -1s), it captures real physics: how materials become magnetic, how phase transitions work, and — fascinatingly — how the brain might store memories.

---

## Part 2: The Grid (Lattice)

```
+1 -1 +1 +1 -1
-1 +1 -1 +1 +1
+1 +1 +1 -1 -1
-1 -1 +1 +1 -1
+1 -1 -1 +1 +1
```

Each cell in the grid is called a **spin** (or lattice site).

In our implementation: `grid[i][j]` = either `+1` or `-1`.

Grid size: **256 × 256** = 65,536 spins total.

---

## Part 3: Energy — Why Do Spins Care About Neighbors?

Here's the key idea: **neighboring spins want to align with each other**.

Two neighbors that are the same (both ↑ or both ↓) = **lower energy** (stable, happy state)  
Two neighbors that are opposite (one ↑, one ↓) = **higher energy** (unstable, unhappy state)

The energy formula for one spin is:
```
E = -J × spin × (sum of 4 neighbors)
```
Where J = 1 (we set it to 1 for simplicity, called "ferromagnetic coupling").

**Example:**
- Spin at center = +1 (UP)
- 3 neighbors = +1 (UP), 1 neighbor = -1 (DOWN)
- Sum of neighbors = 3 + (-1) = 2... wait, let's be precise:

```
     +1
      |
-1 — +1 — +1
      |
     +1
```
Sum of neighbors = (+1) + (+1) + (+1) + (+1) = 4... actually for the example:
```
     +1
      |
-1 — [+1] — +1
      |
     +1
```
neighbors = -1 + 1 + 1 + 1 = 2  
Energy = -1 × (+1) × 2 = -2  (nice and low, stable)

---

## Part 4: Flipping a Spin — The Energy Change (ΔE)

If we flip spin s_ij from +1 to -1 (or vice versa), the change in energy is:

```
ΔE = 2 × spin × (sum of 4 neighbors)
```

**Why?**
- If spin = +1 and all neighbors = +1: ΔE = 2 × 1 × 4 = +8 (flipping costs energy, bad)
- If spin = +1 and all neighbors = -1: ΔE = 2 × 1 × (-4) = -8 (flipping saves energy, good!)

In code:
```c
int neighbors = grid[(i+1)%N][j] + grid[(i-1+N)%N][j]
              + grid[i][(j+1)%N] + grid[i][(j-1+N)%N];
int dE = 2 * spin * neighbors;
```

The `%N` handles **periodic boundary conditions** — the grid wraps around like a torus (donut shape). The right edge connects to the left edge, top connects to bottom.

---

## Part 5: The Metropolis-Hastings Algorithm

Now here's the clever part. Real physics isn't zero temperature — things are warm, and thermal fluctuations happen. Even "bad" moves (that raise energy) can happen occasionally due to random thermal noise.

The **Metropolis-Hastings** algorithm models this:

```
1. Pick a random spin at position (i, j)
2. Calculate ΔE if we were to flip it
3. If ΔE ≤ 0:  ALWAYS flip it (it lowers or keeps energy the same)
4. If ΔE > 0:  flip it with probability exp(-ΔE / T)
```

In code:
```c
if (dE <= 0 || ((double)rand()/RAND_MAX) < exp(-dE / T))
    grid[i][j] = -spin;
```

**The temperature T controls randomness:**
- High T → exp(-ΔE/T) is close to 1 → almost always flip → chaotic, random
- Low T  → exp(-ΔE/T) is close to 0 → rarely flip uphill → ordered, aligned

One "step" = one attempted flip. One "sweep" = N×N attempted flips (one per spin on average).

---

## Part 6: Magnetization — The Output We Measure

```
Magnetization = (sum of all spins) / (total number of spins)
```

- **M = +1.0**: all spins aligned UP (perfectly magnetized)
- **M = -1.0**: all spins aligned DOWN (perfectly magnetized, opposite direction)
- **M = 0.0**: half up, half down (disordered, no net magnetization)

```c
double magnetization() {
    long sum = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += grid[i][j];
    return (double)sum / (N * N);
}
```

At the start (random initialization): M ≈ 0 (expected, random ≈ balanced)  
After many steps at low T: M → ±1 (order emerges)  
After many steps at high T: M stays ≈ 0 (disorder)

---

## Part 7: The Phase Transition — The Most Interesting Part

At a **critical temperature Tc ≈ 2.269** (in natural units), something dramatic happens.

```
T < 2.269:  Spins ORDER into large clusters. M approaches ±1.
T = 2.269:  CRITICAL POINT. Order and disorder coexist at all scales.
T > 2.269:  Spins are DISORDERED. M stays near 0.
```

This is a **phase transition** — just like water turning to ice, except here it's magnetism switching on.

Why is the critical point special?
- The system has patterns at EVERY scale simultaneously
- This is called **scale invariance** or **self-similarity**
- It's the same mathematical structure that appears in fractals
- And — here's the neuroscience connection your proposal mentions — it's the same structure believed to occur in the brain!

That's why we use T = 2.269 in our simulation: it puts the system right at the edge.

---

## Part 8: Why Is This Good for Parallel Computing?

This is the key question for your assignment.

### The Naive (Wrong) Approach
If you try to update all spins at the same time in parallel, you have a **race condition**: thread A is reading spin (1,1)'s neighbors while thread B is updating spin (1,0), which IS one of those neighbors. The result is undefined, incorrect behavior.

### The Red-Black (Checkerboard) Decomposition ✅

Color the grid like a checkerboard:
```
R B R B R B
B R B R B R
R B R B R B
B R B R B R
```
- **R (Red)**: cells where (i+j) is EVEN
- **B (Black)**: cells where (i+j) is ODD

The brilliant observation: **every Red cell's 4 neighbors are all Black cells, and vice versa**.

So:
1. Update ALL Red cells simultaneously in parallel → safe, no races (reading Black, writing Red)
2. Update ALL Black cells simultaneously in parallel → safe, no races (reading Red, writing Black)

This is exactly what all three implementations do:

| Technology | How it parallelizes |
|------------|---------------------|
| **OpenMP** | `#pragma omp parallel for` over Red cells, then Black cells |
| **MPI** | Each process owns rows of the grid; halo exchange for boundary rows |
| **CUDA** | One GPU thread per spin; 65,536 threads update simultaneously |

### Locality (Why It Scales Well)
Each spin only needs its 4 **immediate** neighbors. This means:
- Very little data needs to be shared between parallel workers
- Communication overhead is low (only boundary rows for MPI)
- Cache behavior is good (neighbors are nearby in memory)

---

## Part 9: The Three Technologies Explained Simply

### OpenMP — Shared Memory Parallelism
```
[Your CPU has 8 cores]
Core 1: handles rows 0-31
Core 2: handles rows 32-63
Core 3: handles rows 64-95
...all at the same time, all reading/writing the SAME grid in RAM
```
- Think of it like multiple chefs in one kitchen sharing the same counter
- Easy to implement: add `#pragma omp` directives
- Limited to one machine

### MPI — Distributed Memory Parallelism
```
[Imagine 4 computers connected by network]
Computer 0: owns rows 0-63   [its own private copy]
Computer 1: owns rows 64-127 [its own private copy]
Computer 2: owns rows 128-191
Computer 3: owns rows 192-255

Before each sweep: each computer sends its boundary row to neighbors
```
- Think of it like different kitchens connected by a conveyor belt
- More complex: explicit communication needed
- Scales to thousands of machines (supercomputers use this)

### CUDA — GPU Parallelism
```
[Your RTX 4050 has 2560 CUDA cores]
Core 0:    handles spin (0,0)
Core 1:    handles spin (0,1)
...
Core 65535: handles spin (255,255)
All 65,536 spins updated simultaneously!
```
- Think of it like having 65,536 tiny workers
- Each is slow individually, but massive parallelism wins
- Best for regular, repetitive work (exactly like our grid)

---

## Part 10: Performance Concepts for Your Report

### Speedup
```
Speedup(N) = Time_serial / Time_parallel(N)
```
Example: Serial = 10s, Parallel with 4 cores = 3s → Speedup = 10/3 = 3.33×

### Efficiency
```
Efficiency(N) = Speedup(N) / N × 100%
```
Example: Speedup 3.33 with 4 cores → Efficiency = 83.25%

Perfect efficiency = 100% (rarely achieved due to overhead).

### Amdahl's Law — Why Speedup Has a Limit
Not all code can be parallelized. If 5% must run serially:
```
Max Speedup = 1 / (0.05 + 0.95/N)
With N → ∞: Max Speedup = 1/0.05 = 20×
```
Even with infinite cores, you can only get 20× faster.

### Why Our Speedup Won't Be Ideal
1. **OpenMP**: Thread creation overhead, cache coherency traffic, `exp()` per spin is expensive
2. **MPI**: Halo exchange communication takes time; more processes = more communication
3. **CUDA**: Data transfer between CPU and GPU (PCIe bus) is the bottleneck

---

## Part 11: The Neuroscience Connection (From Your Proposal)

This is why you found this algorithm "genuinely meaningful":

**Hopfield Networks** (1982, John Hopfield — Nobel Prize 2024!) are directly inspired by the Ising Model.

| Ising Model | Hopfield Network |
|-------------|-----------------|
| Spin (+1/-1) | Neuron (firing/not firing) |
| Low-energy state | Stored memory pattern |
| Metropolis update | Neural update rule |
| Critical temperature | Optimal recall temperature |

The brain is believed to operate near its own critical point — just like our T = 2.269 simulation. This means it's at the boundary between too much order (rigid, can't learn) and too much chaos (random, can't remember).

This isn't just a metaphor — the mathematics is identical.

---

## Part 12: Your Assignment Checklist

### Code You Need to Submit
- [ ] `serial/ising_serial.c` — baseline (provided, working)
- [ ] `openmp/ising_openmp.c` — parallel with threads
- [ ] `mpi/ising_mpi.c` — parallel with processes
- [ ] `cuda/ising_cuda.cu` — parallel on GPU
- [ ] All Makefiles

### Data You Need to Collect (Run on Your Machine)
- [ ] Serial execution time (baseline)
- [ ] OpenMP: times for 1, 2, 4, 8, 16 threads
- [ ] MPI: times for 1, 2, 4, 8, 16 processes
- [ ] CUDA: times for block sizes 8×8, 16×16, 32×32

### What to Put in Your Report (3-4 pages)
1. **Parallelization Strategy** — explain Red-Black decomposition, how each tech uses it
2. **Runtime Configuration** — your hardware, compiler versions, commands used
3. **Performance Analysis** — speedup graphs, efficiency, bottlenecks
4. **Critical Reflection** — what was hard, what you'd do differently

### Submission Checklist
- [ ] ZIP with Serial/, OpenMP/, MPI/, CUDA/ subdirectories + Makefiles + README
- [ ] Report PDF (3-4 pages, IEEE format references)
- [ ] Email approval PDF (the email chain with Prof. Nuwan)
- [ ] 3-minute video recording of all 4 implementations running
