#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define BLOCK_DIM 10
#define max_function(a,b) ((a)>(b)?(a):(b)) 
struct __align__(8) MD//引入MD结构体，同时更新最大值和全局求和
{
    float max_tmp;//负责存储最大值
    float sum_tmp;//负责存储求和
};
__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.max_tmp > b.max_tmp);
    MD bigger = a_bigger ? a : b;
    MD smaller = a_bigger ? b : a;
    MD res;
    res.sum_tmp = bigger.sum_tmp + smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp);
    res.max_tmp = bigger.max_tmp;
    return res;
}
double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
__global__
void softmax_kernel(float *input, int size){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    MD md_partial;
    md_partial.max_tmp = -__FLT_MAX__;
    md_partial.sum_tmp = 0.0f;
    
    for(int id = threadIdx.x; id < size; id += BLOCK_DIM){
        MD md_input;
        md_input.max_tmp = input[id];
        md_input.sum_tmp = 1.0f;
        md_partial = reduce_md_op(md_partial, md_input);//每隔一个线程块就做一个比较，把所有信息集中到一个线程块
    } 
    typedef cub::BlockReduce<MD, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_block = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (threadIdx.x == 0){//必须指定threadIdx.x = 0来写入全局内存
        md_total = md_block;
    }
    __syncthreads();
    //-----------------
    float max_total, sum_inverse_total;
    max_total = md_total.max_tmp;
    sum_inverse_total = __fdividef(1.0F, md_total.sum_tmp);
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




