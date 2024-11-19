#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Swap function...
void swap(int *x, int *y)
{
int temp = *x;
*x = *y;
*y = temp;
}

// Partition function for quicksort..
int partition(int *nums, int low, int high)
{
int i = low, j = high;
int pivot = low;  
while (i < j) 
{
while (nums[i] <= nums[pivot] && i < high)
{
i++;
}
while (nums[j] > nums[pivot] && j > low)
{
j--;
}
if (i < j)
{
swap(&nums[i], &nums[j]);
}
}
swap(&nums[pivot], &nums[j]); 
return j;
}

// Quicksort function...
void quickSort(int *nums, int low, int high)
{
if (low < high)  
{
int p = partition(nums, low, high);
quickSort(nums, low, p - 1);
quickSort(nums, p + 1, high);
}
}

 
void parallelQuickSort(int *nums, int n, int rank, int size, MPI_Comm comm)
{
int k = n / size;
int remainder = n % size;
int local_size = k + (rank < remainder ? 1 : 0);  
int *local_nums = (int *)malloc(local_size * sizeof(int));
int *sendcounts = (int *)malloc(size * sizeof(int));
int *displs = (int *)malloc(size * sizeof(int));
int sum = 0;
for (int i = 0; i < size; i++)
{
sendcounts[i] = k + (i < remainder ? 1 : 0);
displs[i] = sum;
sum += sendcounts[i];
}
MPI_Scatterv(nums, sendcounts, displs, MPI_INT, local_nums, local_size, MPI_INT, 0, comm);
quickSort(local_nums, 0, local_size - 1);
MPI_Gatherv(local_nums, local_size, MPI_INT, nums, sendcounts, displs, MPI_INT, 0, comm);

if (rank == 0)
{
// In-place merging of sorted arrays from different processes...
int *temp = (int *)malloc(n * sizeof(int));
int l, r, t, mid;
for (int i = 1; i < size; ++i)
{
mid = displs[i]; 
// Starting index of the next chunk to merge
int end = mid + sendcounts[i];
int idx = 0;
for (l = 0, r = mid; l < mid && r < end;)
{
if (nums[l] < nums[r])
temp[idx++] = nums[l++];
else
temp[idx++] = nums[r++];
}
while (l < mid)
temp[idx++] = nums[l++];
while (r < end)
temp[idx++] = nums[r++];
for (t = 0; t < idx; t++)
nums[t] = temp[t];
}
free(temp);
}
free(local_nums);
free(sendcounts);
free(displs);
}

int main(int argc, char **argv)
{
int rank, size;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int *nums = NULL;
int n;
if (rank == 0)
{
FILE *inputFile = fopen("input_qs.txt", "r");
if (!inputFile)
{
fprintf(stderr, "Error opening input file\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
fscanf(inputFile, "%d", &n); 
nums = (int *)malloc(n * sizeof(int));
for (int i = 0; i < n; i++)
{
fscanf(inputFile, "%d", &nums[i]);
}

fclose(inputFile);
}
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (rank != 0)
{
nums = (int *)malloc(n * sizeof(int)); 
// Resize the nums array in other processes
}

parallelQuickSort(nums, n, rank, size, MPI_COMM_WORLD);
if (rank == 0)
{
FILE *outputFile = fopen("output_qs.txt", "w");
if (!outputFile)
{
fprintf(stderr, "Error opening output file\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}

fprintf(outputFile, "Sorted Array: ");
for (int i = 0; i < n; i++)
{
fprintf(outputFile, "%d ", nums[i]);
}
fprintf(outputFile, "\n");

fclose(outputFile);
free(nums);
}

MPI_Finalize();
return 0;
}
