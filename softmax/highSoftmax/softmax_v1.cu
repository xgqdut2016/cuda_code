#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define BLOCK_DIM_x 2
#define BLOCK_DIM_y 8
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
__global__ void _softmax_kernel(float *__restrict input,
                         int size, int dimsize,
                         int stride) {  // if set axis = 1
    int i = threadIdx.x + blockIdx.x * blockDim.x; // i < inputShape[axis]
    int j = threadIdx.y + blockIdx.y * blockDim.y; // j < size/inputShape[axis]
    int size_x = dimsize;
    int size_y = size/size_x;
    int tid = j % stride + (j - j % stride) * size_x;
    
    MD md_partial;
    md_partial.max_tmp = -__FLT_MAX__;
    md_partial.sum_tmp = 0.0f;
    for (int id = threadIdx.x; id < size_x; id += blockDim.x) {
        MD md_input;
        md_input.max_tmp = input[tid + id * stride];
        md_input.sum_tmp = 1.0f;
        md_partial = reduce_md_op(md_partial,
                                      md_input); // reduce the data to one block
    }
    typedef cub::BlockReduce<MD, BLOCK_DIM_x> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    MD md_block =
            BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    __shared__ float res_sum[BLOCK_DIM_y];
    __shared__ float res_max[BLOCK_DIM_y];
    if (threadIdx.x ==
            0) { // must set threadIdx.x = 0 write the output to memory
        res_sum[threadIdx.y] = md_block.sum_tmp;
        res_max[threadIdx.y] = md_block.max_tmp;
    }
    //__syncthreads();
        //-----------------
    float sum_inverse_total;
        
    sum_inverse_total = __fdividef(1.0F, res_sum[threadIdx.y]);
    if(i < size_x && j < size_y){
        input[tid + i * stride] =
            __expf(input[tid + i * stride] - res_max[threadIdx.y]) * sum_inverse_total;
    }
    
    
}
void softmax(float *cpu_input, int size, int *cpu_inputShape, int axis, int nDims, int stride){
    double st, ela;
    st = get_walltime();
    int size_x = cpu_inputShape[axis];
    int size_y = size/cpu_inputShape[axis];
    int num_block_x = ceil(size_x/(double)BLOCK_DIM_x);
    int num_block_y = ceil(size_y/(double)BLOCK_DIM_y);
    dim3 block_dim(BLOCK_DIM_x,BLOCK_DIM_y,1);
    dim3 grid_dim(num_block_x,num_block_y,1);
    
    
    float *input, *res_sum, *res_max;
    cudaMalloc((void **) &input, size*sizeof(float));
    cudaMalloc((void **) &res_sum, size_y*sizeof(float));
    cudaMalloc((void **) &res_max, size_y*sizeof(float));
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    int *inputShape;
    cudaMalloc((void **) &inputShape, nDims*sizeof(float));
    cudaMemcpy(inputShape, cpu_inputShape, nDims*sizeof(float), cudaMemcpyHostToDevice);
    int share_mem = (2*BLOCK_DIM_y + 2)*sizeof(float); 
    
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    _softmax_kernel<<<grid_dim, block_dim, share_mem>>>(input, size, size_x, stride);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(inputShape);
    cudaFree(res_sum);
    cudaFree(res_max);
    ela = get_walltime() - st;
    
    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
    
}
int main() {
    
    int nDims = 2;
    int cpu_inputShape[nDims] = {2,4};
    int axis = 0;
    
    int size = 1;
    int stride = 1;
    for(int i = nDims - 1; i >= 0; --i){
        if(i == axis){
            stride = size;
        }
        size *= cpu_inputShape[i];
        
    }
    //printf("stride:%d\n",stride);
    
    float cpu_input[size] = {0, 1, 2, 3, 10000, 10001, 10002, 10003};
    softmax(cpu_input, size, cpu_inputShape, axis, nDims, stride);
    
    float s = 0;
    
    for(int i = 0; i < size; i++){
        s += cpu_input[i];
        printf("softmax:%.4e\n",cpu_input[i]);
    }
    printf("s:%.3e\n",s);
    
    
    
    return 0;
}





