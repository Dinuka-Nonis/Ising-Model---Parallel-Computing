#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* grid size, number of sweeps, temperature (critical point is around 2.269) */
#define N     256
#define STEPS 1000
#define T     2.269

static int grid[N][N];

/*
 * precompute the two boltzmann factors we actually need
 * dE can only be +4 or +8 when we need exp(), so just store both
 * index 0 = exp(-4/T), index 1 = exp(-8/T)
 */
static double exp_table[2];

static void build_exp_table(void) {
    exp_table[0] = exp(-4.0 / T);
    exp_table[1] = exp(-8.0 / T);
}

/* returns the right boltzmann factor without calling exp() in the loop */
static inline double boltzmann(int dE) {
    return (dE == 4) ? exp_table[0] : exp_table[1];
}

/* fill grid with random +1 or -1 spins */
static void initialize(void) {
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = (rand() % 2) * 2 - 1;
}

/* average spin value across the whole grid, should be near 0 at start */
static double magnetization(void) {
    long sum = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += grid[i][j];
    return (double)sum / (N * N);
}

/*
 * red-black (checkerboard) sweep - same algorithm used in openmp and mpi
 * versions so timing comparisons are fair
 *
 * pass 0 (red):   update cells where (i+j) is even
 * pass 1 (black): update cells where (i+j) is odd
 *
 * cells of the same color never neighbor each other so updates are independent
 * j_start skips straight to the first cell of the right color in each row
 */
static void metropolis_serial(void) {
    unsigned int seed = 42;

    for (int sweep = 0; sweep < STEPS; sweep++) {
        for (int color = 0; color < 2; color++) {
            for (int i = 0; i < N; i++) {
                /* first column of this color in row i */
                int j_start = ((i % 2) == color) ? 0 : 1;
                for (int j = j_start; j < N; j += 2) {
                    int spin = grid[i][j];

                    /* sum of 4 neighbors, wraps around at edges (periodic boundary) */
                    int nb = grid[(i + 1) % N][j]
                           + grid[(i - 1 + N) % N][j]
                           + grid[i][(j + 1) % N]
                           + grid[i][(j - 1 + N) % N];

                    int dE = 2 * spin * nb;

                    if (dE <= 0) {
                        /* always flip if energy goes down or stays same */
                        grid[i][j] = -spin;
                    } else {
                        /* flip with boltzmann probability if energy goes up */
                        double r = (double)rand_r(&seed) / RAND_MAX;
                        if (r < boltzmann(dE))
                            grid[i][j] = -spin;
                    }
                }
            }
        }
    }
}

int main(void) {
    struct timespec ts, te;

    build_exp_table();
    initialize();

    printf("=== Ising Model - Serial (Red-Black, Unified Baseline) ===\n");
    printf("Grid: %dx%d | Temperature: %.3f | Sweeps: %d\n", N, N, T, STEPS);
    printf("Initial magnetization: %.4f\n", magnetization());

    clock_gettime(CLOCK_MONOTONIC, &ts);
    metropolis_serial();
    clock_gettime(CLOCK_MONOTONIC, &te);

    double elapsed = (te.tv_sec - ts.tv_sec)
                   + (te.tv_nsec - ts.tv_nsec) / 1e9;

    printf("Final magnetization:   %.4f\n", magnetization());
    printf("Execution time: %.4f seconds\n", elapsed);
    return 0;
}
