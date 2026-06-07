#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#define N     256
#define STEPS 1000
#define T     2.269

/*
 * Ghost-row layout:
 *   row 0      = bottom ghost (copy of row N)
 *   rows 1..N  = real grid
 *   row N+1    = top ghost (copy of row 1)
 * Eliminates all vertical % from the inner loop entirely.
 */
static int8_t grid[N+2][N];

static double e4, e8;

/*
 * xorshift32: 3 XOR-shifts, no division, no memory beyond one uint32.
 * ~4 cycles vs rand_r's ~15-20.  Fine for Metropolis MC.
 */
static inline uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}
/* multiply by 2^-32 instead of dividing by RAND_MAX - one FMUL, no FDIV */
#define RAND01(s) (xorshift32(s) * 2.3283064365386963e-10)

static void initialize(void) {
    unsigned int seed = 42;
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = (int8_t)((rand_r(&seed) % 2) * 2 - 1);
}

static double magnetization(void) {
    long sum = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < N; j++)
            sum += grid[i][j];
    return (double)sum / (N * N);
}

static inline void sync_ghosts(void) {
    memcpy(grid[0],   grid[N], N);   /* ghost below = last real row  */
    memcpy(grid[N+1], grid[1], N);   /* ghost above = first real row */
}

static void metropolis_serial(void) {
    uint32_t seed = 1234567891u;

    for (int sweep = 0; sweep < STEPS; sweep++) {
        for (int color = 0; color < 2; color++) {

            sync_ghosts();  /* 2x 256-byte memcpy - negligible */

            for (int i = 1; i <= N; i++) {
                const int8_t *up  = grid[i+1];  /* no % - ghost handles it */
                const int8_t *dn  = grid[i-1];
                int8_t       *cur = grid[i];

                int j_start = (((i-1) & 1) == color) ? 0 : 1;

                for (int j = j_start; j < N; j += 2) {
                    /*
                     * Horizontal wrap: only two sites per row touch the edge.
                     * Use branchless ternary (cmov) instead of %.
                     */
                    int jp1 = (j == N-1) ? 0   : j+1;
                    int jm1 = (j == 0)   ? N-1 : j-1;

                    int spin = cur[j];
                    int nb   = up[j] + dn[j] + cur[jp1] + cur[jm1];
                    int dE   = 2 * spin * nb;

                    if (dE <= 0) {
                        cur[j] = (int8_t)-spin;
                    } else {
                        /* branch-free boltzmann lookup */
                        if (RAND01(&seed) < ((dE == 4) ? e4 : e8))
                            cur[j] = (int8_t)-spin;
                    }
                }
            }
        }
    }
}

int main(void) {
    struct timespec ts, te;
    e4 = exp(-4.0 / T);
    e8 = exp(-8.0 / T);
    initialize();
    printf("=== Ising Model - Serial (optimised) ===\n");
    printf("Grid: %dx%d | T: %.3f | Sweeps: %d\n", N, N, T, STEPS);
    printf("Initial magnetization: %.4f\n", magnetization());
    clock_gettime(CLOCK_MONOTONIC, &ts);
    metropolis_serial();
    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;
    printf("Final magnetization:   %.4f\n", magnetization());
    printf("Execution time: %.4f seconds\n", elapsed);
    return 0;
}