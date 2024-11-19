#include <stdio.h>
#include <math.h>
#include <mpi.h>

double norm(const double* vec, int n) {
    double temp = 0.0;
    for (int i = 0; i < n; i++) {
        temp += vec[i] * vec[i];
    }
    return sqrt(temp);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 4; // Matrix and vector size
    double A[4][4];
    double b[4], X_old[4] = {0}, X_new[4] = {0};
    double tolerance = 0.00001;

    if (rank == 0) {
        double A_init[4][4] = {
            {10, -1, 2, 0},
            {-1, 11, -1, 3},
            {2, -1, 10, -1},
            {0, 3, -1, 8}
        };
        double b_init[4] = {6, 25, -11, 15};

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A_init[i][j];
            }
            b[i] = b_init[i];
        }
    }

    // Calculate how many rows each process will handle
    int k = n / size; 
    double A_local[4][4]; // Adjust the size if you have more than 4 processes
    double b_local[4];
    double A_flat[16];

    // Flatten the matrix A for scattering
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_flat[i * n + j] = A[i][j];
            }
        }
    }

    // Scatter the matrix and vector
    MPI_Scatter(A_flat, k * n, MPI_DOUBLE, A_local, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, k, MPI_DOUBLE, b_local, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(X_old, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double X_new_local[4]; // Size should match k
    while (1) {
        for (int i = 0; i < k; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i * n + j != j) { // Correct the condition to avoid self-reference
                    sum += A_local[i][j] * X_old[j]; // Accessing A_local correctly
                }
            }
            X_new_local[i] = (b_local[i] - sum) / A_local[i][i];
        }

        // Gather results back to X_new
        MPI_Gather(X_new_local, k, MPI_DOUBLE, X_new, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (fabs(norm(X_old, n) - norm(X_new, n)) < tolerance) {
                break; // Check for convergence
            }
            for (int i = 0; i < n; i++) {
                X_old[i] = X_new[i]; // Update old values for the next iteration
            }
        }

        MPI_Bcast(X_old, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Final solution: ");
        for (int i = 0; i < n; i++) {
            printf("%f ", X_old[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
