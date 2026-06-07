#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <mpi.h>

#define N     256
#define STEPS 1000
#define T     2.269   /* critical temperature of the 2D Ising model */

/*
 * Each rank owns a horizontal stripe of rows.
 * We add one ghost row on each side to hold boundary data from neighbours.
 * This avoids needing to communicate during the inner loop.
 *   row 0            -> top ghost    (filled by rank above)
 *   rows 1..local_rows -> real data
 *   row local_rows+1 -> bottom ghost (filled by rank below)
 */
#define G(i,j)  local_grid[((i)+1)*N + (j)]
#define GTOP(j) local_grid[j]
#define GBOT(j) local_grid[(local_rows+1)*N + (j)]

/* fast RNG - avoids the division in rand_r/RAND_MAX */
static inline uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *s = x;
}
#define RAND01(s) (xorshift32(s) * 2.3283064365386963e-10) /* multiply by 2^-32 */

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (N % nprocs != 0) {
        if (rank == 0) fprintf(stderr, "N=%d not divisible by nprocs=%d\n", N, nprocs);
        MPI_Finalize(); return 1;
    }

    /* precompute Boltzmann factors - only dE=4 or dE=8 are possible */
    double e4 = exp(-4.0 / T), e8 = exp(-8.0 / T);
    int local_rows = N / nprocs;

    /* int8_t saves memory and matches the serial version */
    int8_t *local_grid = malloc((local_rows + 2) * N * sizeof(int8_t));

    /*
     * Reproduce the same starting lattice as the serial code (rand_r seed=42).
     * Each rank fast-forwards past the rows belonging to lower ranks, then
     * fills its own stripe - so single-process MPI matches serial exactly.
     */
    unsigned int iseed = 42u;
    for (int k = 0; k < rank * local_rows * N; k++) rand_r(&iseed);
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            G(i,j) = (int8_t)((rand_r(&iseed) % 2) * 2 - 1);

    /* gather initial magnetization across all ranks and print from rank 0 */
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

    /* periodic neighbours: rank 0 wraps to rank nprocs-1 and vice versa */
    int rank_up   = (rank - 1 + nprocs) % nprocs;
    int rank_down = (rank + 1) % nprocs;

    /* each rank gets a unique seed so RNG sequences don't correlate */
    uint32_t seed = 1234567891u + (uint32_t)rank * 2654435761u;

    double t0 = MPI_Wtime();

    for (int sweep = 0; sweep < STEPS; sweep++) {
        /*
         * Checkerboard (red-black) decomposition: update only sites of one
         * color per pass so no two adjacent sites are updated simultaneously.
         * Requires a halo exchange before each color pass.
         */
        for (int color = 0; color < 2; color++) {

            /* non-blocking halo exchange: overlap communication setup with nothing yet,
             * then Waitall ensures ghosts are ready before we touch them */
            MPI_Request reqs[4];
            MPI_Isend(&G(0,0),            N, MPI_INT8_T, rank_up,   0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(&GTOP(0),           N, MPI_INT8_T, rank_up,   1, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(&G(local_rows-1,0), N, MPI_INT8_T, rank_down, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&GBOT(0),           N, MPI_INT8_T, rank_down, 0, MPI_COMM_WORLD, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            for (int i = 0; i < local_rows; i++) {
                /* point directly at neighbour rows - avoids per-site branch in inner loop */
                const int8_t *up  = (i == 0)            ? &GTOP(0) : &G(i-1, 0);
                const int8_t *dn  = (i == local_rows-1) ? &GBOT(0) : &G(i+1, 0);
                int8_t       *cur = &G(i, 0);

                /* global row index needed to determine which color this row starts on */
                int j_start = (((rank * local_rows + i) & 1) == color) ? 0 : 1;

                for (int j = j_start; j < N; j += 2) {
                    int jp1  = (j == N-1) ? 0 : j+1;   /* periodic horizontal wrap */
                    int jm1  = (j == 0)   ? N-1 : j-1;
                    int spin = cur[j];
                    int dE   = 2 * spin * (up[j] + dn[j] + cur[jp1] + cur[jm1]);
                    /* accept flip if energy decreases, or with Boltzmann probability */
                    if (dE <= 0 || RAND01(&seed) < (dE == 4 ? e4 : e8))
                        cur[j] = (int8_t)-spin;
                }
            }
        }
    }

    double t1 = MPI_Wtime();

    /* collect final magnetization from all ranks */
    lsum = 0;
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++) lsum += G(i,j);
    MPI_Reduce(&lsum, &gsum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Final   magnetization: %.4f\n", (double)gsum / (N*N));
        printf("Execution time: %.4f seconds\n", t1 - t0);
    }

    free(local_grid);
    MPI_Finalize();
    return 0;
}