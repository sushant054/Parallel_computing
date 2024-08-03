#include<iostream>
#include<mpi.h>
using namespace std;
int main(int argc, char **argv)
{
int rank;
int size;
int global_sum=0;
int arr[]={1,2,3,4,5,6,7,8,9,10,11,12};

MPI_Init(&argc, &argv);//communication word will be created
MPI_Comm_rank(MPI_COMM_WORLD, &rank);//process id..
MPI_Comm_size(MPI_COMM_WORLD, &size);//no of process

if(rank==0){
int sum=0;
for(int i=0;i<=2;i++)
{
sum+=arr[i];
}
cout<<"sum0:"<<sum<<endl;
//total sum..
int temp=0;
global_sum+=sum;
for(int i=1;i<size;i++){
    MPI_Recv(&temp,1,MPI_INT, i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    global_sum+=temp;
}
cout<<"Sum of all Numbers:"<<global_sum<<endl;
}

if(rank==1){
int sum=0;
for(int i=3;i<=5;i++){
sum+=arr[i];
}
cout<<"sum1:"<<sum<<endl;
MPI_Send(&sum, 1, MPI_INT,0,0,MPI_COMM_WORLD);
}


if(rank==2)
{
int sum=0;
for(int i=6;i<=8;i++)
{
sum+=arr[i];
}

cout<<"sum2:"<<sum<<endl;
MPI_Send(&sum, 1, MPI_INT,0,0,MPI_COMM_WORLD);
}

if(rank==3)
{
int sum=0;
for(int i=9;i<=11;i++){
sum+=arr[i];
}
cout<<"sum3:"<<sum<<endl;
MPI_Send(&sum, 1, MPI_INT,0,0,MPI_COMM_WORLD);
}


MPI_Finalize();//terminates..
return 0;
}
