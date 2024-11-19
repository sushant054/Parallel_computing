#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
int rank, size;
int n, m;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// File pointers
FILE *inputFile;
FILE *outputFile;

if (rank == 0)
{
// Rank 0 opens the input file for reading 
inputFile = fopen("input.txt", "r");
if (inputFile == NULL)
{
printf("Error opening input file.\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
//here only rank 0 reads the dimensions from the file
fscanf(inputFile, "%d %d", &n, &m);
}
//here we send the dimensions to all processes from rank 0
if (rank == 0)
{
for (int i = 1; i < size; i++)
{
MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
MPI_Send(&m, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
}
}
else
{
MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(&m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
int local_n = n / size; 
int local_A[local_n][m];
int B[m];                 
int local_Y[local_n];     
if (rank == 0)
{
int A[n][m];
// Rank 0 reads the matrix from the input file
for (int i = 0; i < n; i++)
{
for (int j = 0; j < m; j++)
{
fscanf(inputFile, "%d", &A[i][j]);
}
}

// Rank 0 reads the vector B from the input file
for (int i = 0; i < m; i++)
{
fscanf(inputFile, "%d", &B[i]);
}
fclose(inputFile);
// Scatter the rows of the matrix to all processes
MPI_Scatter(A, local_n * m, MPI_INT, local_A, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);

// Send vector B to all processes
for (int i = 1; i < size; i++)
{
MPI_Send(B, m, MPI_INT, i, 0, MPI_COMM_WORLD);
}
}
else
{
// Receive the scattered matrix rows from rank 0
MPI_Scatter(NULL, local_n * m, MPI_INT, local_A, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);
// Receive vector B from rank 0
MPI_Recv(B, m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
//compute the local result for Y (Y = A * B)..
for (int i = 0; i < local_n; i++)
{
local_Y[i] = 0;
for (int j = 0; j < m; j++)
{
local_Y[i] += local_A[i][j] * B[j];
}
}
int Y[n]; // result vector in rank 0..
// gather all local Y results into Y in rank 0..
MPI_Gather(local_Y, local_n, MPI_INT, Y, local_n, MPI_INT, 0, MPI_COMM_WORLD);
if (rank == 0)
{
outputFile = fopen("output.txt", "w");
if (outputFile == NULL)
{
printf("Error opening output file.\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
fprintf(outputFile, "The result vector Y is:\n");
for (int i = 0; i < n; i++)
{
fprintf(outputFile, "%d\n", Y[i]);
}
fclose(outputFile);
}
MPI_Finalize();
return 0;
}
//this code showing correct output for 2 and 4 processors..
