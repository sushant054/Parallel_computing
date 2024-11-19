#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
int myrank, nprocs;
int sum = 0, global_sum = 0;
 
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

int array_size = 12;   
int n = array_size / nprocs;
int remainder = array_size % nprocs;
int *A = NULL;
int *B = (int *)malloc(n * sizeof(int));   
if (myrank == 0) {
A = (int *)malloc(array_size * sizeof(int));

FILE *input_file = fopen("input.txt", "r");
if (input_file == NULL) {
printf("Error opening input.txt\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
for (int i = 0; i < array_size; i++) {
fscanf(input_file, "%d", &A[i]);
}
fclose(input_file);
// disc...chunks to other processes
for (int i = 1; i < nprocs; i++) {
int offset = n * i + (i < remainder ? i : remainder);
int count = n + (i < remainder ? 1 : 0);
MPI_Send(&A[offset], count, MPI_INT, i, 0, MPI_COMM_WORLD);
}
// calc..sum of the root's chunk, including any remainder ele..
int start = 0;
int count = n + (0 < remainder ? 1 : 0);
for (int i = 0; i < count; i++) {
sum += A[start + i];
}
global_sum = sum;
// collect sums from other processes
for (int i = 1; i < nprocs; i++) {
int temp = 0;
MPI_Recv(&temp, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
global_sum += temp;
}
//global_sum to output.txt
FILE *output_file = fopen("output.txt", "w");
if (output_file == NULL) {
printf("Error opening output.txt\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
fprintf(output_file, "global_sum is = %d\n", global_sum);
fclose(output_file);
printf("global_sum is = %d\n", global_sum);   
free(A);   
} 
else 
{
 
int count = n + (myrank < remainder ? 1 : 0);
MPI_Recv(B, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
for (int i = 0; i < count; i++) {
sum += B[i];
}
MPI_Send(&sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
}
printf("sum from process %d is %d\n", myrank, sum);
free(B); 

MPI_Finalize();
return 0;
}





