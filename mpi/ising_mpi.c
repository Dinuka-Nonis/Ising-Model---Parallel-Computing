#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N     256
#define STEPS 1000
#define T     2.269

/*
 * each process owns a horizontal stripe of rows plus two ghost rows
 * row 0 in the allocation = top ghost (received from the process above)
 * rows 1..local_rows = actual data this process owns
 * row local_rows+1 = bottom ghost (received from the process below)
 * G(i,j) indexes into the real rows (offset by 1 to skip the top ghost)
 */
#define G(i, j)      local_grid[((i) + 1) * N + (j)]
#define GHOST_TOP(j) local_grid[0 * N + (j)]
#define GHOST_BOT(j) local_grid[(local_rows + 1) * N + (j)]

/* same exp lookup trick as the serial/openmp versions */
static double exp_lut[2];

static void build_exp_table(void) {
    exp_lut[0] = exp(-4.0 / T);
    exp_lut[1] = exp(-8.0 / T);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* grid must divide evenly among processes */
    if (N % nprocs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N=%d must be divisible by nprocs=%d\n", N, nprocs);
        MPI_Finalize();
        return 1;
    }

    build_exp_table();

    int local_rows = N / nprocs;

    /* allocate stripe + 2 ghost rows */
    int *local_grid = (int *)malloc((local_rows + 2) * N * sizeof(int));
    if (!local_grid) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Finalize();
        return 1;
    }

    /* each rank gets a different seed so spins are independent */
    unsigned int seed = 42u + (unsigned)rank * 1000u;
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            G(i, j) = (rand_r(&seed) % 2) * 2 - 1;

    /* wrap-around neighbors for periodic boundary (torus topology) */
    int up   = (rank - 1 + nprocs) % nprocs;
    int down = (rank + 1) % nprocs;

    double t_start = MPI_Wtime();

    for (int sweep = 0; sweep < STEPS; sweep++) {

        /*
         * do halo exchange once per color pass (not once per sweep)
         * after the red pass, boundary rows have new values that the
         * black pass needs to read - so we must exchange again
         */
        for (int color = 0; color < 2; color++) {

            /*
             * non-blocking halo exchange - post all 4 sends/recvs at once
             * tag 10 = sending top row, tag 11 = sending bottom row
             * MPI_Waitall blocks until all 4 finish before we touch ghost rows
             */
            MPI_Request reqs[4];

            MPI_Isend(&G(0, 0),            N, MPI_INT, up,   10, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(local_grid,          N, MPI_INT, up,   11, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(&G(local_rows-1, 0), N, MPI_INT, down, 11, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&G(local_rows, 0),   N, MPI_INT, down, 10, MPI_COMM_WORLD, &reqs[3]);

            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            /* ghost rows are valid now, safe to read them */
            for (int i = 0; i < local_rows; i++) {
                /* global row index needed to figure out which color this row starts on */
                int global_i = rank * local_rows + i;
                int j_start  = (global_i & 1) ^ color;

                for (int j = j_start; j < N; j += 2) {
                    int spin = G(i, j);

                    /* use ghost row macros for boundary rows, no branching needed */
                    int top    = (i == 0)            ? GHOST_TOP(j) : G(i - 1, j);
                    int bottom = (i == local_rows-1) ? GHOST_BOT(j) : G(i + 1, j);
                    int left   = G(i, (j - 1 + N) % N);
                    int right  = G(i, (j + 1) % N);

                    int nb = top + bottom + left + right;
                    int dE = 2 * spin * nb;

                    if (dE <= 0) {
                        G(i, j) = -spin;
                    } else {
                        /* dE/4 - 1 gives index 0 for dE=4, index 1 for dE=8 */
                        double r = (double)rand_r(&seed) / RAND_MAX;
                        if (r < exp_lut[dE / 4 - 1])
                            G(i, j) = -spin;
                    }
                }
            }
        } /* color loop */
    } /* sweep loop */

    double t_end = MPI_Wtime();

    /* sum all local spins, collect result at rank 0 */
    long local_sum = 0;
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            local_sum += G(i, j);

    long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double mag = (double)global_sum / (N * N);
        printf("=== Ising Model - MPI (Parallel) ===\n");
        printf("Processes: %d | Grid: %dx%d | Temperature: %.3f | Sweeps: %d\n",
               nprocs, N, N, T, STEPS);
        printf("Final magnetization: %.4f\n", mag);
        printf("Execution time: %.4f seconds\n", t_end - t_start);
    }

    free(local_grid);
    MPI_Finalize();
    return 0;
}
