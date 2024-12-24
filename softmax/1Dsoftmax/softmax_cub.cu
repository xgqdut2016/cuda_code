#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define BLOCK_DIM 10
#define max_function(a,b) ((a)>(b)?(a):(b))

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
__global__
void max_soft(float *input, int size, float *result){
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size)
    {
        float value = input[tid];
        // 执行规约操作
        float block_max = BlockReduce(temp_storage).Reduce(value,cub::Max());
        // 只有第一个线程将结果写回到全局内存
        if (threadIdx.x == 0)
        {
            result[blockIdx.x] = block_max;
        }
    }
    
}
__global__ void gridmax(float *result, int num_blocks){//result[0] = max
    for(int stride = num_blocks/2; stride > 0; stride = stride/2){
        if (threadIdx.x < stride) {//threadIdx.x + stride < num_blocks = len(result)
            result[threadIdx.x] = max_function(result[threadIdx.x + stride], result[threadIdx.x]);
        }
        __syncthreads();
    }
}

__global__
void sum_soft(float *input, int size, float *result){
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size)
    {
        input[tid] = __expf(input[tid] - result[0]);
        float value = input[tid];
        // 执行规约操作
        float block_sum = BlockReduce(temp_storage).Reduce(value,cub::Sum());
        // 只有第一个线程将结果写回到全局内存
        if (threadIdx.x == 0)
        {
            result[blockIdx.x] = block_sum;
        }
    }
}
__global__ void gridsum(float *result, int num_blocks){//result[0] = sum
    for(int stride = num_blocks/2; stride > 0; stride = stride/2){
        if (threadIdx.x < stride) {
            result[threadIdx.x] += result[threadIdx.x + stride];
        }
        __syncthreads();
    }
    //printf("sum:%.3e\n",result[0]);非常奇怪，如果打印sum，结果正确，否则结果错误
}
__global__ void softmax(float *input, float *result, int size){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size){
        input[tid] /= result[0];
    }
}
void cpu_softmax(float *cpu_input, int size){
    double st, ela;
    st = get_walltime();
    
    int num_blocks = ceil(size/(double)BLOCK_DIM);
    dim3 block_dim(BLOCK_DIM,1,1);
    dim3 grid_dim(num_blocks,1,1);
    
    int mem_size = num_blocks*sizeof(float);
    float *input, *result, *cpu_result;
    cudaMalloc((void **) &input, size*sizeof(float));
    cudaMalloc((void **) &result, mem_size);
    cpu_result = (float *)malloc(mem_size);
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    max_soft<<<grid_dim, block_dim>>>(input, size, result);
    
    gridmax<<<grid_dim, block_dim>>>(result, num_blocks);
    
    sum_soft<<<grid_dim, block_dim>>>(input, size, result);
    
    gridsum<<<grid_dim, block_dim>>>(result, num_blocks);
    
    softmax<<<grid_dim, block_dim>>>(input, result, size);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(result);
    free(cpu_result);
    ela = get_walltime() - st;
    
    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
    
}

int main() {
    float *cpu_input;
    int size = 16;
    
    cpu_input = (float *)malloc(size*sizeof(float));
    for(int i = 0; i < size; i++){
        cpu_input[i] = i%100;
        
    }
    cpu_softmax(cpu_input, size);
    
   
    for(int i = 0; i < size; i++){
        
        printf("softmax:%.4e\n",cpu_input[i]);
    }
    
    free(cpu_input);
    
    
    return 0;
}




