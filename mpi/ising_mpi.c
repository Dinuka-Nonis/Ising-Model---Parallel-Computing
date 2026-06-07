#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <mpi.h>

#define N     256
#define STEPS 1000
#define T     2.269

/*
 * Each rank owns a horizontal stripe of rows.
 * Ghost rows hold boundary data from neighbours.
 *   row 0              -> top ghost    (filled by rank above)
 *   rows 1..local_rows -> real data
 *   row local_rows+1   -> bottom ghost (filled by rank below)
 */
#define G(i,j)  local_grid[((i)+1)*N + (j)]
#define GTOP(j) local_grid[j]
#define GBOT(j) local_grid[(local_rows+1)*N + (j)]

/*
 * Per-site RNG states - CRITICAL for matching serial results.
 * rng_states[(i)*N + j] is the xorshift32 state for local row i, column j.
 * Each site is seeded from its GLOBAL row index so every implementation
 * (serial, OpenMP, MPI with any process count) derives the same state for
 * the same physical site and produces identical physics.
 */

/* Wang hash: mixes bits into a non-zero seed */
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
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *s = x;
}
#define RAND01(s) (xorshift32(s) * 2.3283064365386963e-10)

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (N % nprocs != 0) {
        if (rank == 0) fprintf(stderr, "N=%d not divisible by nprocs=%d\n", N, nprocs);
        MPI_Finalize(); return 1;
    }

    double e4 = exp(-4.0 / T), e8 = exp(-8.0 / T);
    int local_rows = N / nprocs;

    int8_t   *local_grid  = malloc((local_rows + 2) * N * sizeof(int8_t));
    uint32_t *rng_states  = malloc(local_rows * N * sizeof(uint32_t));

    /*
     * Reproduce the same starting lattice as the serial code (rand_r seed=42).
     * Fast-forward past rows belonging to lower ranks, then fill this stripe.
     * Per-site RNG seeded from GLOBAL row index to match serial and OpenMP.
     */
    unsigned int iseed = 42u;
    for (int k = 0; k < rank * local_rows * N; k++) rand_r(&iseed);
    for (int i = 0; i < local_rows; i++) {
        int global_row = rank * local_rows + i;
        for (int j = 0; j < N; j++) {
            G(i,j) = (int8_t)((rand_r(&iseed) % 2) * 2 - 1);
            /* same seeding formula as serial: global_row and column j */
            rng_states[i * N + j] = wang_hash((uint32_t)(global_row * N + j) ^ 0xABCD1234u);
        }
    }

    /* gather initial magnetization */
    long lsum = 0;
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++) lsum += G(i,j);
    long gsum = 0;
    MPI_Reduce(&lsum, &gsum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("=== Ising Model - MPI (optimised) ===\n");
        printf("Processes: %d | Grid: %dx%d | T: %.3f | Sweeps: %d\n", nprocs, N, N, T, STEPS);
        printf("Initial magnetization: %.4f\n", (double)gsum / (N*N));
    }

    int rank_up   = (rank - 1 + nprocs) % nprocs;
    int rank_down = (rank + 1) % nprocs;

    double t0 = MPI_Wtime();

    for (int sweep = 0; sweep < STEPS; sweep++) {
        for (int color = 0; color < 2; color++) {

            /* non-blocking halo exchange */
            MPI_Request reqs[4];
            MPI_Isend(&G(0,0),            N, MPI_INT8_T, rank_up,   0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(&GTOP(0),           N, MPI_INT8_T, rank_up,   1, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(&G(local_rows-1,0), N, MPI_INT8_T, rank_down, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&GBOT(0),           N, MPI_INT8_T, rank_down, 0, MPI_COMM_WORLD, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            for (int i = 0; i < local_rows; i++) {
                const int8_t *up  = (i == 0)            ? &GTOP(0) : &G(i-1, 0);
                const int8_t *dn  = (i == local_rows-1) ? &GBOT(0) : &G(i+1, 0);
                int8_t       *cur = &G(i, 0);
                uint32_t     *rr  = &rng_states[i * N]; /* per-site RNG for row i */

                int global_row = rank * local_rows + i;
                int j_start = ((global_row & 1) == color) ? 0 : 1;

                for (int j = j_start; j < N; j += 2) {
                    int jp1  = (j == N-1) ? 0 : j+1;
                    int jm1  = (j == 0)   ? N-1 : j-1;
                    int spin = cur[j];
                    int dE   = 2 * spin * (up[j] + dn[j] + cur[jp1] + cur[jm1]);

                    if (dE <= 0) {
                        cur[j] = (int8_t)-spin;
                    } else {
                        if (RAND01(&rr[j]) < (dE == 4 ? e4 : e8))
                            cur[j] = (int8_t)-spin;
                    }
                }
            }
        }
    }

    double t1 = MPI_Wtime();

    lsum = 0;
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++) lsum += G(i,j);
    MPI_Reduce(&lsum, &gsum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Final   magnetization: %.4f\n", (double)gsum / (N*N));
        printf("Execution time: %.4f seconds\n", t1 - t0);
    }

    free(local_grid);
    free(rng_states);
    MPI_Finalize();
    return 0;
}
