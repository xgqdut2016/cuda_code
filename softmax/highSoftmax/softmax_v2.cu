#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>


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
template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
void _softmax_kernel(float *input, int size, int *inputShape, int axis, int nDims, int stride){
    int tid = 0;                       // tid = i(JKS) + j(KS) + k(S) + s
    int dimsize = inputShape[axis]; // set axis = 1, dimsize = J
    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                    dimsize; // now, tid = i(JKS) + k(S) + s;

    MD md_partial;
    md_partial.max_tmp = -__FLT_MAX__;
    md_partial.sum_tmp = 0.0f;
    for(int id = threadIdx.x; id < dimsize; id += blockDim.x){
        MD md_input;
        md_input.max_tmp = input[tid + id*stride];
        md_input.sum_tmp = 1.0f;
        md_partial = reduce_md_op(md_partial, md_input);//每隔一个线程块就做一个比较，把所有信息集中到一个线程块
    } 
    typedef cub::BlockReduce<MD, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
        
    __shared__ MD md_total;
    MD md_block = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (threadIdx.x ==  0) { // must set threadIdx.x = 0 write the output to memory
        md_total = md_block;
    }
    __syncthreads();
        //-----------------
    float max_total, sum_inverse_total;
    max_total = md_total.max_tmp;
    sum_inverse_total = __fdividef(1.0F, md_total.sum_tmp);
    for (int id = threadIdx.x; id < dimsize; id += blockDim.x) {
        input[tid + id * stride] =
            __expf(input[tid + id * stride] - max_total) * sum_inverse_total;
    }
}
    

void softmax(float *cpu_input, int size, int *cpu_inputShape, int axis, int nDims){
    double st, ela;
    st = get_walltime();
    int dimsize = cpu_inputShape[axis];
    int num_blocks = size/dimsize;
    float *input;
    cudaMalloc((void **) &input, size*sizeof(float));
    
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    int *inputShape;
    cudaMalloc((void **) &inputShape, nDims*sizeof(float));
    cudaMemcpy(inputShape, cpu_inputShape, nDims*sizeof(float), cudaMemcpyHostToDevice);
    int stride = 1, temp = 1; // stride=[JKS, KS, S, 1][axis]
        
        
    for (int i = nDims - 1; i >= 0; --i) { // must i = nDims - 1, --i; can't i = 0, i++
        if (i == axis) {
            stride = temp;
        }
        temp *= cpu_inputShape[i];
    }
    
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    if (dimsize > 1023) {
        int BLOCK_DIM = 1024;
        _softmax_kernel<1024><<<num_blocks, BLOCK_DIM>>>(
            input, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 511) {
        int BLOCK_DIM = 512;
        _softmax_kernel<512><<<num_blocks, BLOCK_DIM>>>(
            input, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 255) {
        int BLOCK_DIM = 256;
        _softmax_kernel<256><<<num_blocks, BLOCK_DIM>>>(
            input, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 127) {
        int BLOCK_DIM = 128;
        _softmax_kernel<128><<<num_blocks, BLOCK_DIM>>>(
            input, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 63) {
        int BLOCK_DIM = 64;
        _softmax_kernel<64><<<num_blocks, BLOCK_DIM>>>(
            input, size, inputShape, axis, nDims, stride);
    } else {
        int BLOCK_DIM = 32;
        _softmax_kernel<32><<<num_blocks, BLOCK_DIM>>>(
            input, size, inputShape, axis, nDims, stride);
    }
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(inputShape);
    
    ela = get_walltime() - st;
    
    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
    
}
int main() {
    
    int nDims = 2;
    int cpu_inputShape[nDims] = {2,4};
    int axis = 1;
    
    int size = 1;
    for(int i = nDims - 1; i >= 0; --i){
        size *= cpu_inputShape[i];
        
    }
    //printf("stride:%d\n",stride);
    
    float cpu_input[size] = {0, 1, 2, 3, 10000, 10001, 10002, 10003};
    softmax(cpu_input, size, cpu_inputShape, axis, nDims);
    
    float s = 0;
    for(int i = 0; i < size; i++){
        s += cpu_input[i];
        printf("softmax:%.4e\n",cpu_input[i]);
    }
    printf("s:%.3e\n",s);
    
    
    
    return 0;
}




