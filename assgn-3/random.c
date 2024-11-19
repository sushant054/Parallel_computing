#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int n, m;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        // Only rank 0 takes input for matrix dimensions
        printf("Enter the number of rows (n):\n");
        scanf("%d", &n);
        printf("Enter the number of columns (m):\n");
        scanf("%d", &m);
    }

    // Broadcast the dimensions to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;  // Number of rows each process will handle
    int local_A[local_n][m]; // Local chunk of the matrix
    int B[m];                // Vector B (size m)
    int local_Y[local_n];    // Local result vector

    if (rank == 0)
    {
        int A[n][m];

        // Rank 0 takes input for the matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                printf("Enter the value at row %d and column %d: \n", i + 1, j + 1);
                scanf("%d", &A[i][j]);
            }
        }

        // Rank 0 takes input for the vector B
        for (int i = 0; i < m; i++)
        {
            printf("Enter the value at row %d of vector B:\n ", i + 1);
            scanf("%d", &B[i]);
        }

        // Scatter the rows of the matrix to all processes
        MPI_Scatter(A, local_n * m, MPI_INT, local_A, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        // Receive the scattered matrix rows from rank 0
        MPI_Scatter(NULL, local_n * m, MPI_INT, local_A, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Broadcast vector B to all processes
    MPI_Bcast(B, m, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the local result for Y (Y = A * B)
    for (int i = 0; i < local_n; i++)
    {
        local_Y[i] = 0;
        for (int j = 0; j < m; j++)
        {
            local_Y[i] += local_A[i][j] * B[j];
        }
    }

    int Y[n]; // Result vector in rank 0

    // Gather all local Y results into Y in rank 0
    MPI_Gather(local_Y, local_n, MPI_INT, Y, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Print the final result vector
        printf("The result vector Y is:\n");
        for (int i = 0; i < n; i++)
        {
            printf("%d\n", Y[i]);
        }
    }

    MPI_Finalize();
    return 0;
}



