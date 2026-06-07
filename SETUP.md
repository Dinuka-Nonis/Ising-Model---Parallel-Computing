# 🧲 Ising Model — Setup & Run Guide
### SE3082 Parallel Computing | IT23614130

---

## Your Machine Specs (What Matters)
| Component | Your Spec | Why it Matters |
|-----------|-----------|----------------|
| CPU | Intel i5-12450H (8 cores) | OpenMP uses these cores |
| RAM | 16 GB | Grid fits easily in RAM |
| GPU | NVIDIA RTX 4050 (6 GB VRAM) | CUDA runs on this |
| OS | Windows 11, 64-bit | Determines install method |

---

## Step 1 — Install WSL2 (Windows Subsystem for Linux)

Everything runs inside WSL2 (Ubuntu on Windows). This is the standard way to do HPC development on Windows.

Open **PowerShell as Administrator** and run:
```powershell
wsl --install
```
Restart your PC. Then open **Ubuntu** from the Start Menu and create a username/password.

---

## Step 2 — Install Build Tools Inside WSL2

Open Ubuntu terminal and run each block:

### 2a. Update packages
```bash
sudo apt update && sudo apt upgrade -y
```

### 2b. Install GCC and basic tools
```bash
sudo apt install -y gcc g++ make build-essential
```

### 2c. Install OpenMP (already included with GCC, but verify)
```bash
gcc --version
# Should show gcc 11.x or newer
echo '#include <omp.h>
int main(){return 0;}' > /tmp/test_omp.c
gcc -fopenmp /tmp/test_omp.c -o /tmp/test_omp && echo "OpenMP OK"
```

### 2d. Install MPI (Open MPI)
```bash
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
mpicc --version  # Verify installation
mpirun --version
```

---

## Step 3 — Install CUDA Toolkit

This is the most involved step. Your RTX 4050 is an **Ada Lovelace** GPU (sm_89).

### 3a. Install CUDA in WSL2
```bash
# Add NVIDIA package repository for WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

### 3b. Add CUDA to your PATH
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3c. Verify CUDA
```bash
nvcc --version
# Should show: Cuda compilation tools, release 12.4
nvidia-smi
# Should show your RTX 4050 info
```

> **Note**: CUDA in WSL2 requires your Windows NVIDIA driver to be up to date.
> Download from: https://www.nvidia.com/drivers (Game Ready or Studio driver, latest version)

---

## Step 4 — Get the Code Into WSL2

### Option A: Copy from Windows
Your Windows files are accessible in WSL2 at `/mnt/c/Users/YourName/`.
```bash
# Example: if your zip is on Desktop
cp /mnt/c/Users/YourName/Desktop/IT23614130_ising.zip ~/
cd ~
unzip IT23614130_ising.zip
cd IT23614130_ising
```

### Option B: Just create the folder
```bash
mkdir -p ~/ising_project
# Then paste files using your editor of choice
```

---

## Step 5 — Compile All Implementations

### Serial
```bash
cd ~/ising_project/serial
make
# Or manually: gcc -O2 -o ising_serial ising_serial.c -lm
```

### OpenMP
```bash
cd ~/ising_project/openmp
make
# Or: gcc -O2 -fopenmp -o ising_openmp ising_openmp.c -lm
```

### MPI
```bash
cd ~/ising_project/mpi
make
# Or: mpicc -O2 -o ising_mpi ising_mpi.c -lm
```

### CUDA
```bash
cd ~/ising_project/cuda
make
# Or: nvcc -O2 -arch=sm_89 -o ising_cuda ising_cuda.cu -lcurand -lm
# sm_89 = your RTX 4050's compute capability
```

---

## Step 6 — Run & Benchmark

### Serial (baseline)
```bash
cd ~/ising_project/serial
./ising_serial
```

### OpenMP — test different thread counts
```bash
cd ~/ising_project/openmp
./ising_openmp 1    # 1 thread
./ising_openmp 2    # 2 threads
./ising_openmp 4    # 4 threads
./ising_openmp 8    # 8 threads
./ising_openmp 16   # 16 threads (hyperthreads)
# Or run all at once:
make benchmark
```

### MPI — test different process counts
```bash
cd ~/ising_project/mpi
mpirun -np 1 ./ising_mpi
mpirun -np 2 ./ising_mpi
mpirun -np 4 ./ising_mpi
mpirun -np 8 ./ising_mpi
# Or:
make benchmark
```

### CUDA — test different block sizes
```bash
cd ~/ising_project/cuda
./ising_cuda 8    # 8x8   = 64 threads per block
./ising_cuda 16   # 16x16 = 256 threads per block  (recommended)
./ising_cuda 32   # 32x32 = 1024 threads per block
# Or:
make benchmark
```

---

## Step 7 — Record Results for Report

Create a table like this as you run each benchmark:

### OpenMP Results Table
| Threads | Time (s) | Speedup (T1/Tn) |
|---------|----------|-----------------|
| 1       | X.XX     | 1.00            |
| 2       | X.XX     | X.XX            |
| 4       | X.XX     | X.XX            |
| 8       | X.XX     | X.XX            |
| 16      | X.XX     | X.XX            |

**Speedup formula:** `Speedup = Time_with_1_thread / Time_with_N_threads`

### MPI Results Table
| Processes | Time (s) | Speedup |
|-----------|----------|---------|
| 1         | X.XX     | 1.00    |
| 2–16      | ...      | ...     |

### CUDA Results Table
| Block Size | Threads/Block | Time (s) | Speedup vs Serial |
|------------|---------------|----------|--------------------|
| 8×8        | 64            | X.XX     | X.XX               |
| 16×16      | 256           | X.XX     | X.XX               |
| 32×32      | 1024          | X.XX     | X.XX               |

---

## Step 8 — Screenshot Everything

For your submission, take screenshots showing:
1. Compilation command + success message
2. Each run command + its output
3. `nvidia-smi` showing your GPU (for CUDA section)
4. `lscpu` showing your CPU info (for OpenMP/MPI section)

Useful commands for hardware info:
```bash
lscpu                    # CPU details
nvidia-smi               # GPU details  
free -h                  # RAM info
cat /proc/version        # Linux/WSL version
mpirun --version         # MPI version
nvcc --version           # CUDA version
gcc --version            # GCC version
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gcc: command not found` | `sudo apt install gcc` |
| `mpicc: command not found` | `sudo apt install openmpi-bin libopenmpi-dev` |
| `nvcc: command not found` | Check CUDA PATH in `~/.bashrc`, re-run `source ~/.bashrc` |
| CUDA error: no device | Make sure Windows NVIDIA driver is updated |
| `fatal error: omp.h: No such file` | `sudo apt install libgomp1` |
| MPI runs but hangs | Try `mpirun --allow-run-as-root -np 4 ./ising_mpi` |
| CUDA: sm_89 unsupported | Your nvcc version might be old, try `-arch=sm_86` |

---

## Video Recording (3 minutes)

Record your screen in WSL2 showing:
- (0:00–0:30) Brief intro: your name, student ID, algorithm
- (0:30–1:00) Compile all 4 versions
- (1:00–2:00) Run each version, show output
- (2:00–3:00) Show speedup numbers, brief explanation

Use **OBS Studio** or **Windows Game Bar (Win+G)** to record.
