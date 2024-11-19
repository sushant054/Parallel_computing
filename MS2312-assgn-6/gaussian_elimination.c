#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void gaussian_elimination(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows - 1; i++) {
        // Find pivot
        double pivot = fabs(matrix[i][i]);
        int pivot_row = i;
        for (int k = i + 1; k < rows; k++) {
            if (fabs(matrix[k][i]) > pivot) {
                pivot = fabs(matrix[k][i]);
                pivot_row = k;
            }
        }

        // Swap rows if necessary
        if (pivot_row != i) {
            for (int j = 0; j < cols; j++) {
                double temp = matrix[i][j];
                matrix[i][j] = matrix[pivot_row][j];
                matrix[pivot_row][j] = temp;
            }
        }

        // Check for singular matrix
        if (fabs(matrix[i][i]) < 1e-10) {
            printf("Matrix is singular and cannot be solved.\n");
            return;
        }

        // Eliminate column
        for (int j = i + 1; j < rows; j++) {
            double factor = matrix[j][i] / matrix[i][i];
            for (int k = i; k < cols; k++) {
                matrix[j][k] -= factor * matrix[i][k];
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 0, cols = 0;
    double **matrix = NULL;
    double *flat_matrix = NULL;

    if (rank == 0) {
        // Read matrix dimensions and data
        FILE *input_file = fopen("input.txt", "r");
        if (input_file == NULL) {
            fprintf(stderr, "Unable to open input file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        fscanf(input_file, "%d %d", &rows, &cols);
        
        // Allocate memory for matrix
        matrix = (double **)malloc(rows * sizeof(double *));
        flat_matrix = (double *)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            matrix[i] = &flat_matrix[i * cols];
        }

        // Read matrix data
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(input_file, "%lf", &matrix[i][j]);
            }
        }
        fclose(input_file);
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory in non-root processes
    if (rank != 0) {
        matrix = (double **)malloc(rows * sizeof(double *));
        flat_matrix = (double *)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            matrix[i] = &flat_matrix[i * cols];
        }
    }

    // Broadcast matrix data to all processes
    MPI_Bcast(flat_matrix, rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process performs Gaussian elimination on the entire matrix
    // (This could be optimized to divide work among processes)
    gaussian_elimination(matrix, rows, cols);

    // Write results to file from root process
    if (rank == 0) {
        FILE *output_file = fopen("gaussian_elimination_output.txt", "w");
        if (output_file != NULL) {
            fprintf(output_file, "Final matrix:\n");
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    fprintf(output_file, "%lf ", matrix[i][j]);
                }
                fprintf(output_file, "\n");
            }
            fclose(output_file);
            printf("Output sent to the file!\n");
        } else {
            fprintf(stderr, "Unable to open output file!\n");
        }
    }

    // Clean up
    free(flat_matrix);
    free(matrix);

    MPI_Finalize();
    return 0;
}




