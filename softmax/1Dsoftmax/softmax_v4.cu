#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define BLOCK_DIM 10
#define max_function(a,b) ((a)>(b)?(a):(b)) 
__device__
float device_max(float a, float b){
    return (a > b) ? a:b;
}

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
__global__
void softmax_kernel(float *input, int size){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    float max_partial = -__FLT_MAX__;
    for(int id = threadIdx.x; id < size; id += BLOCK_DIM){
        max_partial = device_max(max_partial, input[id]);//每隔一个线程块就做一个比较，把所有信息集中到一个线程块
    } 
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;//<float,..>里面的float表示返回值的类型
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float max_total;
    float block_max = BlockReduce(temp_storage).Reduce(max_partial,device_max);//或者使用cub::Max()
    if (threadIdx.x == 0){//必须指定threadIdx.x = 0来写入全局内存
        max_total = block_max;//max_total是share才能保证即使threadIdx.x !=0时，线程也能获得max_total的取值
    }
    __syncthreads();
    
    //-----------------
    float sum_partial = 0.0f;
    for(int id = threadIdx.x; id < size; id += BLOCK_DIM){
        sum_partial += __expf(input[id] - max_total);//CUDA高精度exp函数，把所有信息集中到一个线程块
    } 
    
    __shared__ float sum_inverse_total;
    float block_sum = BlockReduce(temp_storage).Reduce(sum_partial,cub::Sum());
    if (threadIdx.x == 0){
        sum_inverse_total = __fdividef(1.0F, block_sum);//高精度除法
    }

    __syncthreads();
    input[tid] = __expf(input[tid] - max_total)*sum_inverse_total;

}
void softmax(float *cpu_input, int size){
    double st, ela;
    st = get_walltime();
    
    int num_blocks = ceil(size/(double)BLOCK_DIM);
    dim3 block_dim(BLOCK_DIM,1,1);
    dim3 grid_dim(num_blocks,1,1);
    
    
    float *input;
    cudaMalloc((void **) &input, size*sizeof(float));
    
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    int share_size = 2*sizeof(float);
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    softmax_kernel<<<grid_dim, block_dim, share_size>>>(input, size);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
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
    softmax(cpu_input, size);
    
    float s = 0;
    for(int i = 0; i < size; i++){
        s += cpu_input[i];
        printf("softmax:%.4e\n",cpu_input[i]);
    }
    printf("s:%.3e\n",s);
    free(cpu_input);
    
    
    return 0;
}




