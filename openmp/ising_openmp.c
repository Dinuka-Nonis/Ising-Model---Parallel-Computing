#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#define N     256
#define STEPS 1000
#define T     2.269

/*
 * Ghost-row layout (identical to serial):
 *   row 0      = bottom ghost (copy of row N)
 *   rows 1..N  = real grid
 *   row N+1    = top ghost  (copy of row 1)
 */
static int8_t  grid[N+2][N];

/*
 * Per-site RNG states - CRITICAL for matching serial results.
 * Each site (i,j) uses its own xorshift32 state seeded from its coordinates.
 * Because the checkerboard coloring guarantees no two adjacent sites are
 * updated in the same color pass, each site's RNG state is only ever
 * accessed by one thread at a time - no race condition, no lock needed.
 * Every thread reads the same state for a given site that the serial code
 * would, so the physics is identical regardless of thread count.
 */
static uint32_t rng[N+2][N];

static double e4, e8;

/* Wang hash: non-linear mix to derive a non-zero seed from coordinates */
static inline uint32_t wang_hash(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return x ? x : 1u;
}

static inline uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x <<  5;
    return *s = x;
}
#define RAND01(s) (xorshift32(s) * 2.3283064365386963e-10)

static void initialize(void) {
    unsigned int seed = 42;
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < N; j++) {
            grid[i][j] = (int8_t)((rand_r(&seed) % 2) * 2 - 1);
            /* same seeding formula as serial: global_row = i-1 */
            rng[i][j] = wang_hash((uint32_t)((i-1) * N + j) ^ 0xABCD1234u);
        }
}

static double magnetization(void) {
    long sum = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < N; j++)
            sum += grid[i][j];
    return (double)sum / (N * N);
}

static inline void sync_ghosts(void) {
    memcpy(grid[0],   grid[N], N);
    memcpy(grid[N+1], grid[1], N);
}

static void metropolis_omp(int nthreads) {
    #pragma omp parallel num_threads(nthreads)
    {
        int tid      = omp_get_thread_num();
        int rows_per = (N + nthreads - 1) / nthreads;
        int i0       = 1 + tid * rows_per;
        int i1       = i0 + rows_per;
        if (i1 > N + 1) i1 = N + 1;

        for (int sweep = 0; sweep < STEPS; sweep++) {
            for (int color = 0; color < 2; color++) {

                /* One thread updates ghost rows; barrier makes them visible to all */
                if (tid == 0)
                    sync_ghosts();
                #pragma omp barrier

                for (int i = i0; i < i1; i++) {
                    const int8_t *up      = grid[i+1];
                    const int8_t *dn      = grid[i-1];
                    int8_t       *cur     = grid[i];
                    uint32_t     *rng_row = rng[i];   /* per-site RNG for this row */

                    int j_start = (((i-1) & 1) == color) ? 0 : 1;

                    for (int j = j_start; j < N; j += 2) {
                        int jp1 = (j == N-1) ? 0   : j+1;
                        int jm1 = (j == 0)   ? N-1 : j-1;

                        int spin = cur[j];
                        int nb   = up[j] + dn[j] + cur[jp1] + cur[jm1];
                        int dE   = 2 * spin * nb;

                        if (dE <= 0) {
                            cur[j] = (int8_t)-spin;
                        } else {
                            if (RAND01(&rng_row[j]) < ((dE == 4) ? e4 : e8))
                                cur[j] = (int8_t)-spin;
                        }
                    }
                }

                #pragma omp barrier
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 4;

    e4 = exp(-4.0 / T);
    e8 = exp(-8.0 / T);

    initialize();

    printf("=== Ising Model - OpenMP (optimised) ===\n");
    printf("Threads: %d | Grid: %dx%d | T: %.3f | Sweeps: %d\n",
           nthreads, N, N, T, STEPS);
    printf("Initial magnetization: %.4f\n", magnetization());

    double t0 = omp_get_wtime();
    metropolis_omp(nthreads);
    double t1 = omp_get_wtime();

    printf("Final   magnetization: %.4f\n", magnetization());
    printf("Execution time: %.4f seconds\n", t1 - t0);
    return 0;
}
