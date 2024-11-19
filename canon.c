#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int dim = 2;
    int dimv[2] = {2, 2};
    int period[2] = {1, 1};
    int reorder = 1;
    MPI_Comm Cart_Topology;

    // Create a Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, dim, dimv, period, reorder, &Cart_Topology);
    int rank;
    MPI_Comm_rank(Cart_Topology, &rank);

    // Define matrices as 2D arrays
    int A[4][4];
    int B[4][4];
    int C[4][4] = {0};
    int A_local[2][2];
    int B_local[2][2];
    int C_local[2][2] = {0};

    // Initialize matrices A and B on the root process
    if (rank == 0)
    {
        int A_temp[4][4] = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}};
        int B_temp[4][4] = {
            {16, 15, 14, 13},
            {12, 11, 10, 9},
            {8, 7, 6, 5},
            {4, 3, 2, 1}};
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                A[i][j] = A_temp[i][j];
                B[i][j] = B_temp[i][j];
            }
        }
    }

    // Define sendcounts and displacements for scattering
    int sendcounts[4] = {4, 4, 4, 4};
    int displs[4] = {0, 2, 8, 10};

    // Scatter submatrices of A and B
    MPI_Scatterv(A, sendcounts, displs, MPI_INT, A_local, 4, MPI_INT, 0, Cart_Topology);
    MPI_Scatterv(B, sendcounts, displs, MPI_INT, B_local, 4, MPI_INT, 0, Cart_Topology);

    // Shift and multiply in steps
    for (int step = 0; step < 2; ++step)
    {
        if (step == 0 && (rank == 2 || rank == 3))
        {
            MPI_Request send_request, recv_request;
            int left, right;
            MPI_Cart_shift(Cart_Topology, 1, -1, &left, &right);
            MPI_Isend(A_local, 4, MPI_INT, left, 0, Cart_Topology, &send_request);
            MPI_Irecv(A_local, 4, MPI_INT, right, 0, Cart_Topology, &recv_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        }

        if (step == 0 && (rank == 1 || rank == 3))
        {
            MPI_Request send_request, recv_request;
            int up, down;
            MPI_Cart_shift(Cart_Topology, 0, 1, &up, &down);
            MPI_Isend(B_local, 4, MPI_INT, up, 1, Cart_Topology, &send_request);
            MPI_Irecv(B_local, 4, MPI_INT, down, 1, Cart_Topology, &recv_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < 2; ++k)
                {
                    C_local[i][j] += A_local[i][k] * B_local[k][j];
                }
            }
        }
    }

    // Gather results into C
    MPI_Gatherv(C_local, 4, MPI_INT, C, sendcounts, displs, MPI_INT, 0, Cart_Topology);

    // Print the result matrix C on the root process
    if (rank == 0)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}





