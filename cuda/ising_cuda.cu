#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define N     256
#define STEPS 1000
#define T     2.269f

__device__ __inline__
uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* convert the random uint to a float between 0 and 1 */
__device__ __inline__
float xorshift_uniform(uint32_t *state) {
    return (float)(xorshift32(state) >> 8) / (float)(1u << 24);
}

/* give each thread a unique starting RNG state using a hash of its index */
__global__ void init_rng_kernel(uint32_t *states, uint32_t base_seed) {
    int tx  = blockIdx.x * blockDim.x + threadIdx.x;
    int ty  = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= N/2 || ty >= N) return;
    int idx = ty * (N/2) + tx;

    /* wang hash to mix the index into a good starting state */
    uint32_t s = (uint32_t)idx ^ base_seed;
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s ^= s >> 4;
    s *= 0x27d4eb2du;
    s ^= s >> 15;
    /* xorshift breaks if state is 0 */
    states[idx] = (s == 0u) ? 1u : s;
}


__global__ void metropolis_kernel(int8_t *grid, uint32_t *states,
                                   int color, float inv_T) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;  /* half-column index */
    int ty = blockIdx.y * blockDim.y + threadIdx.y;  /* row index */

    if (tx >= N/2 || ty >= N) return;

    /* map half-column tx to the actual column j for this color */
    int j_start = (ty + color) & 1;
    int j = j_start + 2 * tx;

    int idx = ty * (N/2) + tx;
    uint32_t rng_state = states[idx];

    int sidx = ty * N + j;

    /* read current spin and its 4 neighbors through read-only cache */
    int spin   = (int)__ldg(&grid[sidx]);
    int top    = (int)__ldg(&grid[((ty - 1 + N) % N) * N + j]);
    int bottom = (int)__ldg(&grid[((ty + 1) % N)     * N + j]);
    int left   = (int)__ldg(&grid[ty * N + (j - 1 + N) % N]);
    int right  = (int)__ldg(&grid[ty * N + (j + 1) % N]);

    int nb = top + bottom + left + right;
    int dE = 2 * spin * nb;


    if (dE <= 0) {
        grid[sidx] = (int8_t)(-spin);
    } else {
        float r = xorshift_uniform(&rng_state);
        if (r < expf(-(float)dE * inv_T))
            grid[sidx] = (int8_t)(-spin);
    }

    /* write rng state back so the next kernel call continues from here */
    states[idx] = rng_state;
}

/* runs on cpu to compute final magnetization after copying grid back */
static double compute_magnetization(int8_t *grid_host) {
    long sum = 0;
    for (int i = 0; i < N * N; i++) sum += (int)grid_host[i];
    return (double)sum / (N * N);
}

int main(int argc, char *argv[]) {
    int BLOCK_SIZE = 16;
    if (argc > 1) BLOCK_SIZE = atoi(argv[1]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== Ising Model - CUDA (Parallel) ===\n");
    printf("Grid: %dx%d | Temperature: %.3f | Sweeps: %d\n", N, N, T, STEPS);
    printf("Block size: %dx%d | GPU: %s | SMs: %d\n",
           BLOCK_SIZE, BLOCK_SIZE, prop.name, prop.multiProcessorCount);

    /* host memory */
    size_t grid_bytes  = N * N * sizeof(int8_t);
    size_t state_bytes = (N/2) * N * sizeof(uint32_t);

    int8_t *h_grid = (int8_t *)malloc(grid_bytes);
    /*
     * use rand_r(seed=42) to match the serial/openmp/mpi initial grid exactly.
     * the global rand() call that was here before produced a different sequence,
     * making the initial magnetization printed here inconsistent with the other runs.
     */
    unsigned int init_seed = 42u;
    for (int i = 0; i < N * N; i++)
        h_grid[i] = (int8_t)((rand_r(&init_seed) % 2) * 2 - 1);

    printf("Initial magnetization: %.4f\n", compute_magnetization(h_grid));

    /* allocate device memory and copy initial grid over */
    int8_t   *d_grid;
    uint32_t *d_states;
    cudaMalloc(&d_grid,   grid_bytes);
    cudaMalloc(&d_states, state_bytes);
    cudaMemcpy(d_grid, h_grid, grid_bytes, cudaMemcpyHostToDevice);

    /*
     * 2D thread grid - x covers N/2 columns (one color at a time)
     *                  y covers all N rows
     * each thread handles exactly one spin per kernel call
     */
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N/2 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N   + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /*
     * initialise rng states before timing starts.
     *
     * the seed 0xDEADBEEF is a fixed constant intentionally.  this means
     * every run of this program, regardless of block size, starts from
     * the same physical state and produces the same final magnetization.
     * the point of varying block size is to measure TIMING differences
     * (occupancy, warp efficiency, SM utilisation), not physics differences.
     * identical magnetization across block sizes therefore confirms that
     * the simulation is deterministic and block size is purely a performance
     * tuning parameter with no effect on correctness.
     *
     * if you want each run to start from a different random state, replace
     * the constant with (uint32_t)time(NULL).
     */
    const uint32_t RNG_SEED = 0xDEADBEEFu;
    init_rng_kernel<<<blocks, threads>>>(d_states, RNG_SEED);
    cudaDeviceSynchronize();

    /* use cuda events for accurate gpu timing */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    float inv_T = 1.0f / T;

    for (int sweep = 0; sweep < STEPS; sweep++) {
        /* red pass - kernels in the same stream run in order so no sync needed */
        metropolis_kernel<<<blocks, threads>>>(d_grid, d_states, 0, inv_T);
        /* black pass - guaranteed to run after red finishes (same stream) */
        metropolis_kernel<<<blocks, threads>>>(d_grid, d_states, 1, inv_T);
    }

    /* single sync at the end instead of syncing every sweep */
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

    /* copy result back and compute final magnetization on cpu */
    cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost);

    printf("Final magnetization:   %.4f\n", compute_magnetization(h_grid));
    printf("Execution time: %.4f seconds\n", elapsed_ms / 1000.0f);

    /* cleanup */
    cudaFree(d_grid);
    cudaFree(d_states);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free(h_grid);

    return 0;
}
