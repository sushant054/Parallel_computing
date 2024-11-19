//matrix vector multiplication
#include <bits/stdc++.h>
#include <mpi.h>
using  namespace std;

int  main(int argc , char** argv)
{
int rank ,size;
MPI_Init(&argc , &argv);
MPI_Comm_size(MPI_COMM_WORLD , &size);
MPI_Comm_rank(MPI_COMM_WORLD , &rank);


int n;
vector<vector<int>> matrix = {{5,67,8,2},{3,4,8,7},{1,2,30,7},{3,6,7,5}};
vector<int> vec={3,6,7,5};



cout<<"Enter the size of the matrix :: " <<endl;
cin>>n;
cout<<"Enter matrix elements :: " <<endl;
for(int i =0 ; i <n ; i++){
for( int j = 0 ;j< n;j++){
cout<<"Element "<<i+1<<j+1<<" :: ";
}

}

cout<<"Enter vector X elements :: "<<endl;
for( int i  = 0 ; i < n ; i++){
cout<<"Element "<<i+1<< " :: ";
cin>>vec[i];
}

int num_rows_per_proc=n/size;
vector<vector<int>> temp_mat(num_rows_per_proc , vector<int> (n)); 	
vector<int> temp_vec;


//SEND MATRIX ROW WISE TO EACH PROCESSOR
MPI_Scatter(&matrix,num_rows_per_proc*n,MPI_INT,&temp_mat,num_rows_per_proc , MPI_INT ,0 , MPI_COMM_WORLD);
MPI_Scatter(&vec , num_rows_per_proc , MPI_INT ,&temp_vec , num_rows_per_proc , MPI_INT , 0 , MPI_COMM_WORLD);

//MPI ALL GATHER TO GATHER THE VECTOR ON ALL THE PROCESSORS
MPI_Allgather(&vec , n, MPI_INT ,&temp_vec , n , MPI_INT,MPI_COMM_WORLD);


vector<int>local_Y;
//matrix-vector multiplication
for(int i =0 ; i < num_rows_per_proc ;i++){
local_Y[i] = 0;
for(int j = 0 ; j < n ;j++){
	local_Y[i] += temp_mat[i][j] * temp_vec[j]; 
}
}

vector<int> ans_y;
if(rank ==0 ){

MPI_Gather(&local_Y , num_rows_per_proc , MPI_INT, &ans_y, num_rows_per_proc,MPI_INT , 0 ,MPI_COMM_WORLD);
for( int i =0;i<n;i++){
cout<<"y"<<i+1<<" :: "<<ans_y[i]<<endl;
}
}


MPI_Finalize();
return 0;
}


// #include <bits/stdc++.h>
// #include <mpi.h>
// using namespace std;

// int main(int argc, char** argv) {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     int n;
//     vector<vector<int>> matrix;
//     vector<int> vec;

//     if (rank == 0) {
//         // Reading matrix and vector from input.txt
//         ifstream infile("input.txt");
//         infile >> n;  // First line: size of the matrix and vector

//         matrix.resize(n, vector<int>(n));
//         vec.resize(n);

//         for (int i = 0; i < n; i++) {
//             for (int j = 0; j < n; j++) {
//                 infile >> matrix[i][j];
//             }
//         }

//         for (int i = 0; i < n; i++) {
//             infile >> vec[i];
//         }
//         infile.close();
//     }

//     // Broadcast the size of the matrix/vector to all processes
//     MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int num_rows_per_proc = n / size;
//     vector<vector<int>> temp_mat(num_rows_per_proc, vector<int>(n));
//     vector<int> temp_vec(n);

//     // Scatter the rows of the matrix
//     MPI_Scatter(&matrix[0][0], num_rows_per_proc * n, MPI_INT, &temp_mat[0][0], num_rows_per_proc * n, MPI_INT, 0, MPI_COMM_WORLD);

//     // Broadcast the vector to all processes
//     MPI_Bcast(&vec[0], n, MPI_INT, 0, MPI_COMM_WORLD);

//     vector<int> local_Y(num_rows_per_proc, 0);

//     // Perform matrix-vector multiplication
//     for (int i = 0; i < num_rows_per_proc; i++) {
//         local_Y[i] = 0;
//         for (int j = 0; j < n; j++) {
//             local_Y[i] += temp_mat[i][j] * vec[j];
//         }
//     }

//     vector<int> ans_y(n);

//     // Gather results from all processes to the root
//     MPI_Gather(&local_Y[0], num_rows_per_proc, MPI_INT, &ans_y[0], num_rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         // Writing the result to output.txt
//         ofstream outfile("output.txt");
//         for (int i = 0; i < n; i++) {
//             outfile << "y" << i + 1 << " :: " << ans_y[i] << endl;
//         }
//         outfile.close();
//     }

//     MPI_Finalize();
//     return 0;
// }
