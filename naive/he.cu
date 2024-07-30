#include <cuda.h>
#include <stdio.h>
__global__ void printHello()
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 计算全局索引
    printf("hello world from GPU by thread:%d\n", index);
}
int main()
{
    dim3 grid_dim = {1, 1, 1};             // 设置线程网格
    dim3 block_dim = {4, 1, 1};            // 设置对应线程块大小
    printHello<<<grid_dim, block_dim>>>(); // 执行kernel
    // int num_blocks = 1;                    // 如果仅仅使用一维线程网格，可以直接指定线程块数目
    // int BLOCK_DIM = 4;                     // 如果仅仅使用一维线程网格，可以直接指定每个线程块的线程数目
    // printHello<<<num_blocks, BLOCK_DIM>>>();
    cudaDeviceSynchronize(); // 如果没有这一行，无法执行打印过程
    return 0;
}