#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double norm(double* vec, int size) {
    double temp = 0.0;
    for (int i = 0; i < size; i++) {
        temp += vec[i] * vec[i];
    }
    return sqrt(temp);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 4;
    int k = n / size;

    double A[4][4] = {{0}};
    double A_flat[16] = {0};
    double A_local[16] = {0};

    double b[4] = {0};
    double b_local[4] = {0};

    double X_old[4] = {0};
    double X_new[4] = {0};

    if (rank == 0) {
        double A_temp[4][4] = {{10, -1, 2, 0}, {-1, 11, -1, 3}, {2, -1, 10, -1}, {0, 3, -1, 8}};
        double b_temp[4] = {6, 25, -11, 15};

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A_temp[i][j];
                A_flat[i * n + j] = A[i][j];
            }
            b[i] = b_temp[i];
        }
    }

    MPI_Scatter(A_flat, k * n, MPI_DOUBLE, A_local, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, k, MPI_DOUBLE, b_local, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(X_old, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double tolerance = 0.00001;
    double X_new_local[4] = {0};

    for (int it = 0; it < 100; it++) {
        for (int i = 0; i < k; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i * n + rank * k + i != j) {
                    X_new_local[i] += A_local[i * n + j] * X_old[j];
                }
            }
            X_new_local[i] = (b_local[i] - X_new_local[i]) / A_local[i * n + rank * k + i];
        }

        MPI_Allgather(X_new_local, k, MPI_DOUBLE, X_new, k, MPI_DOUBLE, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\nX_old: ");
            for (int i = 0; i < n; i++) {
                printf("%f ", X_old[i]);
            }
            printf("\nX_new: ");
            for (int i = 0; i < n; i++) {
                printf("%f ", X_new[i]);
            }
            printf("\n");
            for (int i = 0; i < n; i++) {
                X_old[i] = X_new[i];
            }
        }
    }

    if (rank == 0) {
        printf("\nFinal answer: ");
        for (int i = 0; i < n; i++) {
            printf("%f  ", X_old[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
