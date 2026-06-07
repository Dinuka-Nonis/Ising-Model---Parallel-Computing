#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define N     256
#define STEPS 1000
#define T     2.269f

/* ------------------------------------------------------------------ */
/* RNG - identical algorithm and seeding as serial/OpenMP/MPI          */
/* ------------------------------------------------------------------ */

__device__ __inline__
uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ __inline__
float xorshift_uniform(uint32_t *state) {
    return (float)xorshift32(state) * 2.3283064365386963e-10f;
}

/*
 * Wang hash: same function used in serial/OpenMP/MPI to seed per-site RNG.
 * Every site (global_row, j) gets the same starting state across all
 * implementations so the physics is identical.
 */
__device__ __inline__
uint32_t wang_hash_dev(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return x ? x : 1u;
}

/* ------------------------------------------------------------------ */
/* Init RNG kernel: seed each site's state from its coordinates        */
/* ------------------------------------------------------------------ */
__global__ void init_rng_kernel(uint32_t *states) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;  /* half-column */
    int ty = blockIdx.y * blockDim.y + threadIdx.y;  /* row */
    if (tx >= N/2 || ty >= N) return;

    /*
     * Reconstruct the actual column for this thread from its half-column index.
     * We can't know color here, so seed BOTH columns from each tx position.
     * Thread (tx, ty) covers column j = 2*tx and j = 2*tx+1.
     * We store a state for every site: index = ty*N + j.
     */
    for (int parity = 0; parity < 2; parity++) {
        int j   = 2 * tx + parity;
        int idx = ty * N + j;
        /* same formula as CPU: global_row = ty, column = j */
        states[idx] = wang_hash_dev((uint32_t)(ty * N + j) ^ 0xABCD1234u);
    }
}

/* ------------------------------------------------------------------ */
/* Metropolis kernel                                                    */
/* ------------------------------------------------------------------ */
__global__ void metropolis_kernel(int8_t *grid, uint32_t *states,
                                   int color, float inv_T) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;  /* half-column */
    int ty = blockIdx.y * blockDim.y + threadIdx.y;  /* row */

    if (tx >= N/2 || ty >= N) return;

    /* Map half-column to actual column for this color pass */
    int j_start = (ty + color) & 1;
    int j = j_start + 2 * tx;

    /* Load this site's RNG state */
    int idx = ty * N + j;
    uint32_t rng_state = states[idx];

    int sidx = ty * N + j;

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

    states[idx] = rng_state;
}

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

    size_t grid_bytes  = N * N * sizeof(int8_t);
    size_t state_bytes = N * N * sizeof(uint32_t);  /* one state per site */

    int8_t *h_grid = (int8_t *)malloc(grid_bytes);
    /* same rand_r(seed=42) initialisation as serial/OpenMP/MPI */
    unsigned int init_seed = 42u;
    for (int i = 0; i < N * N; i++)
        h_grid[i] = (int8_t)((rand_r(&init_seed) % 2) * 2 - 1);

    printf("Initial magnetization: %.4f\n", compute_magnetization(h_grid));

    int8_t   *d_grid;
    uint32_t *d_states;
    cudaMalloc(&d_grid,   grid_bytes);
    cudaMalloc(&d_states, state_bytes);
    cudaMemcpy(d_grid, h_grid, grid_bytes, cudaMemcpyHostToDevice);

    /*
     * 2D thread grid:
     *   x covers N/2 half-columns (one color per kernel launch)
     *   y covers all N rows
     */
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N/2 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N   + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /*
     * Full-grid RNG init: seed every site from its (row, col) coordinates
     * using the same Wang hash formula as serial/OpenMP/MPI.
     * Each thread seeds TWO sites (both parities of its column).
     */
    dim3 init_blocks((N/2 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (N   + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_rng_kernel<<<init_blocks, threads>>>(d_states);
    cudaDeviceSynchronize();

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    float inv_T = 1.0f / T;

    for (int sweep = 0; sweep < STEPS; sweep++) {
        metropolis_kernel<<<blocks, threads>>>(d_grid, d_states, 0, inv_T);
        metropolis_kernel<<<blocks, threads>>>(d_grid, d_states, 1, inv_T);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

    cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost);

    printf("Final magnetization:   %.4f\n", compute_magnetization(h_grid));
    printf("Execution time: %.4f seconds\n", elapsed_ms / 1000.0f);

    cudaFree(d_grid);
    cudaFree(d_states);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free(h_grid);

    return 0;
}
