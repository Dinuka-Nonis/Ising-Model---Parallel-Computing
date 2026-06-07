#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N           256
#define STEPS       1000
#define T           2.269
#define MAX_THREADS 64

/*
 * pad each thread's RNG seed to a full cache line (64 bytes)
 * without this, all seeds sit in the same cache line and cores
 * constantly invalidate each other's cache - called false sharing
 * unsigned int is 4 bytes so we pad with 60 extra bytes
 */
typedef struct {
    unsigned int val;
    char _pad[60];
} aligned_seed_t;

static int grid[N][N];

/*
 * only two possible dE values need exp(): +4 and +8
 * precompute both instead of calling exp() inside the hot loop
 * saves millions of expensive exp() calls over 1000 sweeps
 */
static double exp_lut[2];

static void build_exp_table(void) {
    exp_lut[0] = exp(-4.0 / T);
    exp_lut[1] = exp(-8.0 / T);
}

static void initialize(void) {
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = (rand() % 2) * 2 - 1;
}

/* parallel reduction over the whole grid to get magnetization */
static double magnetization(int num_threads) {
    long sum = 0;
    #pragma omp parallel for num_threads(num_threads) reduction(+:sum) schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += grid[i][j];
    return (double)sum / (N * N);
}

/*
 * main simulation - uses one persistent parallel region for all sweeps
 * instead of forking/joining threads every pass (which would be 2000 times)
 * threads fork once here and stay alive until all STEPS are done
 */
static void metropolis_openmp(int num_threads) {

    aligned_seed_t seeds[MAX_THREADS];
    for (int t = 0; t < num_threads; t++)
        seeds[t].val = 42u + (unsigned)t * 1337u;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        /* each thread keeps its seed as a local variable - no sharing at all */
        unsigned int seed = seeds[tid].val;

        for (int sweep = 0; sweep < STEPS; sweep++) {

            /* --- red pass: cells where (i+j) is even --- */
            /*
             * schedule(static) gives each thread a contiguous block of rows
             * contiguous rows = contiguous memory = better cache behavior
             */
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                /* first red column in this row */
                int j_start = (i % 2 == 0) ? 0 : 1;
                for (int j = j_start; j < N; j += 2) {
                    int spin = grid[i][j];
                    int nb   = grid[(i + 1) % N][j]
                             + grid[(i - 1 + N) % N][j]
                             + grid[i][(j + 1) % N]
                             + grid[i][(j - 1 + N) % N];
                    int dE = 2 * spin * nb;

                    if (dE <= 0) {
                        grid[i][j] = -spin;
                    } else {
                        /* table lookup: dE=4 -> index 0, dE=8 -> index 1 */
                        double bolt = exp_lut[dE / 4 - 1];
                        double r    = (double)rand_r(&seed) / RAND_MAX;
                        if (r < bolt)
                            grid[i][j] = -spin;
                    }
                }
            }
            /* implicit barrier here - all threads must finish red before black starts */

            /* --- black pass: cells where (i+j) is odd --- */
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                int j_start = (i % 2 == 0) ? 1 : 0;
                for (int j = j_start; j < N; j += 2) {
                    int spin = grid[i][j];
                    int nb   = grid[(i + 1) % N][j]
                             + grid[(i - 1 + N) % N][j]
                             + grid[i][(j + 1) % N]
                             + grid[i][(j - 1 + N) % N];
                    int dE = 2 * spin * nb;

                    if (dE <= 0) {
                        grid[i][j] = -spin;
                    } else {
                        double bolt = exp_lut[dE / 4 - 1];
                        double r    = (double)rand_r(&seed) / RAND_MAX;
                        if (r < bolt)
                            grid[i][j] = -spin;
                    }
                }
            }
            /* implicit barrier: black done, safe to start next sweep */

        } /* end sweep loop */

        /* write seed back in case we want to check reproducibility later */
        seeds[tid].val = seed;

    } /* threads join here, only once after all sweeps */
}

int main(int argc, char *argv[]) {
    int num_threads = 4;
    if (argc > 1) num_threads = atoi(argv[1]);
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    build_exp_table();
    initialize();

    printf("=== Ising Model - OpenMP (Parallel) ===\n");
    printf("Threads: %d | Grid: %dx%d | Temperature: %.3f | Sweeps: %d\n",
           num_threads, N, N, T, STEPS);
    printf("Initial magnetization: %.4f\n", magnetization(num_threads));

    double t_start = omp_get_wtime();
    metropolis_openmp(num_threads);
    double t_end = omp_get_wtime();

    printf("Final magnetization:   %.4f\n", magnetization(num_threads));
    printf("Execution time: %.4f seconds\n", t_end - t_start);
    return 0;
}
