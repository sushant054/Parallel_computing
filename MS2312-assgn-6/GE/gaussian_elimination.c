#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define MASTER 0

// Function to print matrix to file or stdout
void print_matrix(double** matrix, int rows, int cols, FILE* fp) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            fprintf(fp, "%.6f ", matrix[i][j]);
        }
        fprintf(fp, "\n");
    }
}

// Function to create input file if it doesn't exist
void create_input_file(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if(!fp) {
        printf("Error creating input file\n");
        exit(1);
    }
    // Write the system of equations
    fprintf(fp, "4 5\n");  // 4 rows, 5 columns (including RHS)
    fprintf(fp, "1 2 1 4 13\n");    // a + 2b + c + 4d = 13
    fprintf(fp, "2 0 4 3 28\n");    // 2a + 4c + 3d = 28
    fprintf(fp, "4 2 2 1 20\n");    // 4a + 2b + 2c + d = 20
    fprintf(fp, "-3 1 3 2 6\n");    // -3a + b + 3c + 2d = 6
    fclose(fp);
    printf("Created input file: %s\n", filename);
}

// Function to read matrix from file
int read_matrix(const char* filename, double** matrix, int* rows, int* cols) {
    FILE* fp = fopen(filename, "r");
    if(!fp) {
        printf("Cannot open input file: %s\n", filename);
        return 0;
    }
    fscanf(fp, "%d %d", rows, cols);
    
    for(int i = 0; i < *rows; i++) {
        for(int j = 0; j < *cols; j++) {
            fscanf(fp, "%lf", &matrix[i][j]);
        }
    }
    fclose(fp);
    return 1;
}

int main(int argc, char** argv) {
    int rank, size, rows = 4, cols = 5;
    double **matrix = NULL, **local_matrix = NULL;
    double *pivot_row = NULL;
    int local_rows;
    const char* input_file = "input.txt";
    const char* output_file = "output.txt";
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Master process handles file I/O
    if(rank == MASTER) {
        // Allocate memory for full matrix
        matrix = (double**)malloc(rows * sizeof(double*));
        for(int i = 0; i < rows; i++) {
            matrix[i] = (double*)malloc(cols * sizeof(double));
        }

        // Try to read input file, create it if doesn't exist
        if(!read_matrix(input_file, matrix, &rows, &cols)) {
            printf("Creating new input file...\n");
            create_input_file(input_file);
            if(!read_matrix(input_file, matrix, &rows, &cols)) {
                printf("Fatal error: Cannot read or create input file\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        printf("Input Matrix:\n");
        print_matrix(matrix, rows, cols, stdout);
    }
    
    // Broadcast dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    // Calculate local rows for each process
    local_rows = rows / size;
    if(rank < rows % size) local_rows++;
    
    // Allocate memory for local matrix
    local_matrix = (double**)malloc(local_rows * sizeof(double*));
    for(int i = 0; i < local_rows; i++) {
        local_matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    
    // Arrays for scattering data
    int *sendcounts = NULL, *displs = NULL;
    if(rank == MASTER) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for(int i = 0; i < size; i++) {
            sendcounts[i] = (rows / size + (i < rows % size ? 1 : 0)) * cols;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // Flatten matrix for scattering
    double *flat_matrix = NULL;
    double *flat_local = (double*)malloc(local_rows * cols * sizeof(double));
    
    if(rank == MASTER) {
        flat_matrix = (double*)malloc(rows * cols * sizeof(double));
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                flat_matrix[i * cols + j] = matrix[i][j];
            }
        }
    }
    
    // Scatter data to all processes
    MPI_Scatterv(flat_matrix, sendcounts, displs, MPI_DOUBLE,
                 flat_local, local_rows * cols, MPI_DOUBLE,
                 MASTER, MPI_COMM_WORLD);
    
    // Convert flat array to 2D array
    for(int i = 0; i < local_rows; i++) {
        for(int j = 0; j < cols; j++) {
            local_matrix[i][j] = flat_local[i * cols + j];
        }
    }
    
    // Allocate pivot row
    pivot_row = (double*)malloc(cols * sizeof(double));
    
    // Main Gaussian Elimination Loop
    for(int k = 0; k < rows - 1; k++) {
        // Determine which process owns the pivot row
        int pivot_owner = k / (rows / size);
        if(k % (rows / size) >= sendcounts[pivot_owner] / cols) pivot_owner++;
        
        // Broadcast pivot row
        if(rank == pivot_owner) {
            int local_k = k - displs[pivot_owner] / cols;
            for(int j = 0; j < cols; j++) {
                pivot_row[j] = local_matrix[local_k][j];
            }
        }
        MPI_Bcast(pivot_row, cols, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
        
        // Eliminate in local rows
        for(int i = 0; i < local_rows; i++) {
            int global_i = displs[rank] / cols + i;
            if(global_i > k) {
                double factor = local_matrix[i][k] / pivot_row[k];
                for(int j = k; j < cols; j++) {
                    local_matrix[i][j] -= factor * pivot_row[j];
                }
            }
        }
    }
    
    // Gather results back to master
    MPI_Gatherv(flat_local, local_rows * cols, MPI_DOUBLE,
                flat_matrix, sendcounts, displs, MPI_DOUBLE,
                MASTER, MPI_COMM_WORLD);
    
    if(rank == MASTER) {
        // Convert back to 2D array
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[i][j] = flat_matrix[i * cols + j];
            }
        }
        
        // Write results to output file
        FILE* fp = fopen(output_file, "w");
        if(fp) {
            fprintf(fp, "Final Matrix after Gaussian Elimination:\n");
            print_matrix(matrix, rows, cols, fp);
            
            // Back substitution
            double* solution = (double*)malloc((cols-1) * sizeof(double));
            for(int i = rows - 1; i >= 0; i--) {
                solution[i] = matrix[i][cols-1];
                for(int j = i + 1; j < cols - 1; j++) {
                    solution[i] -= matrix[i][j] * solution[j];
                }
                solution[i] /= matrix[i][i];
            }
            
            fprintf(fp, "\nSolution:\n");
            fprintf(fp, "a = %.6f\n", solution[0]);
            fprintf(fp, "b = %.6f\n", solution[1]);
            fprintf(fp, "c = %.6f\n", solution[2]);
            fprintf(fp, "d = %.6f\n", solution[3]);
            
            fclose(fp);
            printf("Results written to: %s\n", output_file);
            free(solution);
        } else {
            printf("Error: Cannot open output file\n");
        }
    }
    
    // Cleanup
    free(pivot_row);
    free(flat_local);
    for(int i = 0; i < local_rows; i++) {
        free(local_matrix[i]);
    }
    free(local_matrix);
    
    if(rank == MASTER) {
        for(int i = 0; i < rows; i++) {
            free(matrix[i]);
        }
        free(matrix);
        free(flat_matrix);
        free(sendcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}