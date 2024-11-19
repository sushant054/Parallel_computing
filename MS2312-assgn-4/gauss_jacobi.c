#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

//function to calculate the norm of a vector
double norm(double* vec, int n) {
double temp = 0.0;
for (int i = 0; i < n; i++) 
{
temp += vec[i] * vec[i];
}
return sqrt(temp);
}

int main(int argc, char** argv) 
{
int rank, size;
// Initialize MPI
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Initialize variables
int n;
double* A = NULL;
double* b = NULL;
double* A_flat = NULL;
double* A_local = NULL;
double* b_local = NULL;
double* X_old = NULL;
double* X_new_local = NULL;
double* X_new = NULL;
double tolerance = 1e-6;
int max_iterations = 1000;

 // Input the matrix size, matrix A, and vector b from file (done by rank 0)
if (rank == 0) {
FILE *input_file = fopen("input.txt", "r");
if (input_file == NULL) {
printf("Error opening input file.\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}

// Read matrix size n
fscanf(input_file, "%d", &n);

// Allocate memory for matrix A and vector b
A = (double*) malloc(n * n * sizeof(double));
b = (double*) malloc(n * sizeof(double));
A_flat = (double*) malloc(n * n * sizeof(double));

// Read matrix A
for (int i = 0; i < n * n; i++) {
fscanf(input_file, "%lf", &A[i]);
A_flat[i] = A[i];
}

// Read vector b
for (int i = 0; i < n; i++) 
{
fscanf(input_file, "%lf", &b[i]);
}

fclose(input_file);
}

 // Broadcast matrix size n to all processes
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

 // Calculate the number of rows each process will handle
int k = (n + size - 1) / size; // This ensures that we can handle cases where n < size

// Allocate local memory for each process
A_local = (double*) malloc(k * n * sizeof(double));
b_local = (double*) malloc(k * sizeof(double));
X_old = (double*) calloc(n, sizeof(double)); // Initial guess (zero vector)
X_new_local = (double*) calloc(k, sizeof(double));
X_new = (double*) malloc(n * sizeof(double));

// Scatter matrix A and vector b
MPI_Scatter(A_flat, k * n, MPI_DOUBLE, A_local, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Scatter(b, k, MPI_DOUBLE, b_local, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Broadcast initial guess X_old to all processes
MPI_Bcast(X_old, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Gauss-Jacobi iteration
for (int it = 0; it < max_iterations; it++) 
{
// Perform local computation for each process
for (int i = 0; i < k; i++) {
if (rank * k + i < n) { // Only compute if the global index is within bounds
double sum = 0.0;
int global_i = rank * k + i; // Global row index
for (int j = 0; j < n; j++) {
if (global_i != j) 
{ 
// Exclude diagonal element
sum += A_local[i * n + j] * X_old[j];
}
}
X_new_local[i] = (b_local[i] - sum) / A_local[i * n + global_i];
}
}

// Gather all the new X values from each process
MPI_Allgather(X_new_local, k, MPI_DOUBLE, X_new, k, MPI_DOUBLE, MPI_COMM_WORLD);

// Check convergence (only rank 0 performs this check)
if (rank == 0) {
double diff = 0.0;
for (int i = 0; i < n; i++) 
{
diff += fabs(X_new[i] - X_old[i]);
}
// If the difference is less than the tolerance, we have converged
if (diff < tolerance) {
printf("Convergence achieved after %d iterations.\n", it + 1);
break; // Convergence achieved
}
// Update X_old for the next iteration
for (int i = 0; i < n; i++) 
{
X_old[i] = X_new[i];
}
}

// Broadcast the updated X_old to all processes
MPI_Bcast(X_old, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
// Output the result (only rank 0)
if (rank == 0) 
{
FILE *output_file = fopen("output.txt", "w");
if (output_file == NULL) {
printf("Error opening output file.\n");
} 
else
{
fprintf(output_file, "Solution:\n");
for (int i = 0; i < n; i++) 
{    
fprintf(output_file, "%.6f ", X_new[i]);
}
fprintf(output_file, "\n");
fclose(output_file);
printf("Results written to output.txt\n");
}
}

// Free dynamically allocated memory
free(A_local);
free(b_local);
free(X_new_local);
free(X_new);
if (rank == 0)
{
free(A);
free(b);
free(A_flat);
}

MPI_Finalize();
return 0;
}