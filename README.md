# Ising Model — Parallel Computing Assignment
## SE3082 | NONIS P.K.D.T. | IT23614130

### Algorithm: Ising Model / Metropolis–Hastings Spin Simulation
### Domain: Physics Simulations and Computational Modeling

---

## Project Structure
```
IT23614130_ising/
├── README.md          ← This file
├── SETUP.md           ← Step-by-step setup guide for your PC
├── THEORY.md          ← Complete theory explanation (start here!)
├── serial/
│   ├── ising_serial.c
│   └── Makefile
├── openmp/
│   ├── ising_openmp.c
│   └── Makefile
├── mpi/
│   ├── ising_mpi.c
│   └── Makefile
└── cuda/
    ├── ising_cuda.cu
    └── Makefile
```

---

## Quick Start (after setup)

```bash
# Serial
cd serial && make && ./ising_serial

# OpenMP (4 threads)
cd openmp && make && ./ising_openmp 4

# MPI (4 processes)
cd mpi && make && mpirun -np 4 ./ising_mpi

# CUDA (16x16 block size)
cd cuda && make && ./ising_cuda 16
```

---

## Compilation Summary

| Version | Compile Command |
|---------|----------------|
| Serial  | `gcc -O2 -o ising_serial ising_serial.c -lm` |
| OpenMP  | `gcc -O2 -fopenmp -o ising_openmp ising_openmp.c -lm` |
| MPI     | `mpicc -O2 -o ising_mpi ising_mpi.c -lm` |
| CUDA    | `nvcc -O2 -arch=sm_89 -o ising_cuda ising_cuda.cu -lcurand -lm` |

---

## Files to Read First
1. **THEORY.md** — Understand the Ising Model, Metropolis algorithm, Red-Black decomposition
2. **SETUP.md** — Install WSL2, GCC, OpenMPI, CUDA on your Windows machine
3. Source files — Well-commented code explaining every parallelization decision
