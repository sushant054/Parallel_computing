#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 8 // Matrix size
#define BLOCK_SIZE 4 // Block size (for simplicity, assume matrix size is divisible by this)

void printMatrixToFile(int matrix[N][N], int n, FILE *file) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
}

void readMatrixFromFile(int matrix[N][N], FILE *file) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(file, "%d", &matrix[i][j]);
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) { // Ensure we have exactly 4 processes
        if (rank == 0) {
            printf("This program requires 4 processors.\n");
        }
        MPI_Finalize();
        return 0;
    }

    int A[N][N], B[N][N], C[N][N] = {0}; // Matrices

    if (rank == 0) {
        // Root process reads the input matrices from file
        FILE *input_file = fopen("input.txt", "r");
        if (input_file == NULL) {
            printf("Error opening input file.\n");
            MPI_Finalize();
            return -1;
        }

        // Read matrices A and B from the file
        readMatrixFromFile(A, input_file);
        readMatrixFromFile(B, input_file);
        fclose(input_file);
    }

    int local_A[BLOCK_SIZE][BLOCK_SIZE], local_B[BLOCK_SIZE][BLOCK_SIZE], local_C[BLOCK_SIZE][BLOCK_SIZE] = {0};

    // Scatter sub-blocks of matrix A and B among processes
    MPI_Scatter(A, BLOCK_SIZE * BLOCK_SIZE, MPI_INT, local_A, BLOCK_SIZE * BLOCK_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, BLOCK_SIZE * BLOCK_SIZE, MPI_INT, local_B, BLOCK_SIZE * BLOCK_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Local block matrix multiplication
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < BLOCK_SIZE; k++) {
                local_C[i][j] += local_A[i][k] * local_B[k][j];
            }
        }
    }

    // Gather results into matrix C at root process
    MPI_Gather(local_C, BLOCK_SIZE * BLOCK_SIZE, MPI_INT, C, BLOCK_SIZE * BLOCK_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process writes the result to the output file
    if (rank == 0) {
        FILE *output_file = fopen("output.txt", "w");
        if (output_file == NULL) {
            printf("Error opening output file.\n");
            MPI_Finalize();
            return -1;
        }

        // Write the result matrix to the output file
        fprintf(output_file, "Result matrix C = A * B:\n");
        printMatrixToFile(C, N, output_file);
        fclose(output_file);
    }

    MPI_Finalize();
    return 0;
}
