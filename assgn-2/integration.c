#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double f_x(double x) {
return sin(x);
}

double booles_rule(double a, double b, int n) {
double h = (b - a) / n;
double sum = 0.0;
for (int i = 0; i < n / 4; i++) {
double x0 = a + 4 * i * h;
double x1 = x0 + h;
double x2 = x1 + h;
double x3 = x2 + h;
double x4 = x3 + h;
sum += (7 * f_x(x0) + 32 * f_x(x1) + 12 * f_x(x2) + 32 * f_x(x3) + 7 * f_x(x4));
}
return (2 * h / 45) * sum;
}

int main(int argc, char** argv) {
int rank, size;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Define the domain and no. of partitions..
double a = 0, b = 1;
int n = 100; 
// ensuring n is a multiple of 4
if (n % 4 != 0) {
if (rank == 0) {
printf("n should be a multiple of 4 for Boole's method.\n");
}

MPI_Finalize();
return -1;
}

// Calculate local range and number of intervals
int local_n = n / size;
double local_a = a + rank * local_n * (b - a) / n;
double local_b = local_a + local_n * (b - a) / n;
if (rank == size - 1) {
local_b = b;
}

 // Calculate the local integral
double local_result = booles_rule(local_a, local_b, local_n);

// Reduce all local integrals..
double global_result = 0.0;
MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
printf("final result using Boole's method: %lf\n", global_result);
FILE *output_file = fopen("output.txt", "w");
if (output_file == NULL) {
printf("Error opening output.txt\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}

fprintf(output_file, "final result using Boole's method: %lf\n", global_result);
fclose(output_file);
}

MPI_Finalize();
return 0;
}
