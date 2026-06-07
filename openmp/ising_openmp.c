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
 * Eliminates all vertical % from the inner loop entirely.
 */
static int8_t grid[N+2][N];

static double e4, e8;

/* ------------------------------------------------------------------ */
/* xorshift32: same as serial - 3 XOR-shifts, no division             */
/* ------------------------------------------------------------------ */
static inline uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x <<  5;
    return *s = x;
}

/* multiply by 2^-32 instead of dividing by RAND_MAX - one FMUL, no FDIV */
#define RAND01(s) (xorshift32(s) * 2.3283064365386963e-10)

/* ------------------------------------------------------------------ */
/* initialize - identical to serial                                    */
/* ------------------------------------------------------------------ */
static void initialize(void) {
    unsigned int seed = 42;
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = (int8_t)((rand_r(&seed) % 2) * 2 - 1);
}

/* ------------------------------------------------------------------ */
/* magnetization - identical to serial                                 */
/* ------------------------------------------------------------------ */
static double magnetization(void) {
    long sum = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < N; j++)
            sum += grid[i][j];
    return (double)sum / (N * N);
}

/* ------------------------------------------------------------------ */
/* sync_ghosts - identical to serial                                   */
/* Only called by a single thread (outside the parallel region) so no */
/* synchronisation overhead; alternatively called inside with a single */
/* designated thread before each barrier.                              */
/* ------------------------------------------------------------------ */
static inline void sync_ghosts(void) {
    memcpy(grid[0],   grid[N], N);   /* ghost below = last real row  */
    memcpy(grid[N+1], grid[1], N);   /* ghost above = first real row */
}

/* ------------------------------------------------------------------ */
/* metropolis_omp                                                      */
/*                                                                     */
/* Checkerboard (red-black) decomposition, same coloring rule as the  */
/* serial code:                                                        */
/*   color 0 -> j_start = 0 when (i-1) is even, else 1               */
/*   color 1 -> j_start = 1 when (i-1) is even, else 0               */
/*                                                                     */
/* Thread decomposition: static row bands, same as the OpenMP draft.  */
/*                                                                     */
/* Ghost-row strategy inside the parallel region:                     */
/*   - One designated thread (tid==0) calls sync_ghosts().            */
/*   - A single #pragma omp barrier follows; every thread then reads  */
/*     fresh ghost rows for the entire color pass.                     */
/*   - This gives exactly 2 barriers per sweep (one per color), same  */
/*     logical count as the original OpenMP draft, but now the ghosts */
/*     are correct.                                                    */
/* ------------------------------------------------------------------ */
static void metropolis_omp(int nthreads) {
    #pragma omp parallel num_threads(nthreads)
    {
        int tid      = omp_get_thread_num();
        int rows_per = (N + nthreads - 1) / nthreads;
        int i0       = 1 + tid * rows_per;          /* first real row for this thread */
        int i1       = i0 + rows_per;
        if (i1 > N + 1) i1 = N + 1;                /* clamp to last real row + 1     */

        /*
         * Per-thread seed: derive from a fixed base the same way the serial
         * code uses a single seed, but offset per thread so threads don't
         * produce correlated sequences.
         */
        uint32_t seed = 1234567891u + (uint32_t)tid * 2654435761u; /* Knuth mult */

        for (int sweep = 0; sweep < STEPS; sweep++) {

            for (int color = 0; color < 2; color++) {

                /* ---- sync ghost rows before each color pass ---- */
                /* Only one thread does the memcpy; the barrier that
                 * follows makes the result visible to all threads.  */
                if (tid == 0)
                    sync_ghosts();
                #pragma omp barrier   /* all threads wait for fresh ghosts */

                /* ---- sweep this thread's row band ---- */
                for (int i = i0; i < i1; i++) {
                    const int8_t *up  = grid[i+1];  /* ghost row handles wrap - no % */
                    const int8_t *dn  = grid[i-1];
                    int8_t       *cur = grid[i];

                    /* same coloring formula as serial */
                    int j_start = (((i-1) & 1) == color) ? 0 : 1;

                    for (int j = j_start; j < N; j += 2) {
                        /* branchless horizontal wrap - identical to serial */
                        int jp1 = (j == N-1) ? 0   : j+1;
                        int jm1 = (j == 0)   ? N-1 : j-1;

                        int spin = cur[j];
                        int nb   = up[j] + dn[j] + cur[jp1] + cur[jm1];
                        int dE   = 2 * spin * nb;

                        if (dE <= 0) {
                            cur[j] = (int8_t)-spin;
                        } else {
                            /* branch-free Boltzmann lookup - identical to serial */
                            if (RAND01(&seed) < ((dE == 4) ? e4 : e8))
                                cur[j] = (int8_t)-spin;
                        }
                    }
                }

                /* ---- barrier before next color pass ---- */
                #pragma omp barrier
            }
        }
    }
}

/* ------------------------------------------------------------------ */
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