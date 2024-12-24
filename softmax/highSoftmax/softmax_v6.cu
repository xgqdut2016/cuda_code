#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
#define warpSize 32
#define max_function(a, b) ((a) > (b) ? (a) : (b))

struct __align__(8) MD
{                  // update the global max and sum, store the output at
                   // max_tmp and sum_tmp
    float max_tmp; // store max
    float sum_tmp; // store sum
};
__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.max_tmp > b.max_tmp);
    MD bigger = a_bigger ? a : b;
    MD smaller = a_bigger ? b : a;
    MD res;
    res.sum_tmp = bigger.sum_tmp +
                  smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp);
    res.max_tmp = bigger.max_tmp;
    return res;
}
template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__ void _blockSoftmaxKernel(
    float *__restrict input, float *__restrict output, int size, int dimsize,
    int stride)
{ // if set axis = 1, inputShape=[I,J,K,S]
  // tid = i(JKS) + j(KS) + k(S) + s

    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    int tid =
        blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                  dimsize; // now, tid = i(JKS) + k(S) + s;

    MD md_partial;
    md_partial.max_tmp = -__FLT_MAX__;
    md_partial.sum_tmp = 0.0f;
    MD md_input;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++)
    {

        md_input.max_tmp = input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride];

        md_input.sum_tmp = 1.0f;
        md_partial = reduce_md_op(md_partial,
                                  md_input); // reduce the data to one block
    }
    typedef cub::BlockReduce<MD, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;
    MD md_block = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (threadIdx.x ==
        0)
    { // must set threadIdx.x = 0 write the output to memory
        md_total = md_block;
    }
    __syncthreads();
    //-----------------

    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++)
    {
        output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
            __expf(input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                   md_total.max_tmp) *
            __fdividef(1.0F, md_total.sum_tmp);
    }
}

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return max(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width = warpSize>
__inline__ __device__ T WarpAllReduce(T val)
{
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__global__ void _warpSoftmaxKernel(float *__restrict input,
                                   float *__restrict output, int size,
                                   int dimsize, int stride)
{
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int otherSize = size / dimsize;
    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    // int num_per_thread = (dimsize + warpSize - 1) / warpSize;
    float buff[32];
    if (otherIdx < otherSize)
    {
        for (int ph = 0; threadIdx.x + ph * warpSize < dimsize; ph++)
        {
            buff[ph] = input[tid + (threadIdx.x + ph * warpSize) * stride];
        }
    }
    __shared__ float max_total[warpSize];
    __shared__ float sum_total[warpSize];
    float max_data = -__FLT_MAX__;
    if (otherIdx < otherSize)
    {
        for (int ph = 0; threadIdx.x + ph * warpSize < dimsize; ph++)
        {
            max_data = max_function(max_data, buff[ph]);
        }
    }

    max_data = WarpAllReduce<MaxOp, float, warpSize>(max_data);

    if (threadIdx.x == 0)
        max_total[threadIdx.y] = max_data;
    //__syncthreads();
    //--------------------------------------------
    float sum_data = 0.0f;
    if (otherIdx < otherSize)
    {
        for (int ph = 0; threadIdx.x + ph * warpSize < dimsize; ph++)
        {
            sum_data += __expf(buff[ph] - max_total[threadIdx.y]);
        }
    }
    sum_data = WarpAllReduce<SumOp, float, warpSize>(sum_data);

    if (threadIdx.x == 0)
        sum_total[threadIdx.y] = sum_data;
    //__syncthreads();
    //--------------------------------------------
    if (otherIdx < otherSize)
    {
        for (int ph = 0; threadIdx.x + ph * warpSize < dimsize; ph++)
        {
            output[tid + (threadIdx.x + ph * warpSize) * stride] =
                __expf(buff[ph] - max_total[threadIdx.y]) *
                __fdividef(1.0F, sum_total[threadIdx.y]);
        }
    }
}

void softmax(float *cpu_input, int size, int *cpu_inputShape, int axis, int nDims)
{
    double st, ela;
    st = get_walltime();
    int dimsize = cpu_inputShape[axis];
    int num_blocks = size / dimsize;
    float *input;
    cudaMalloc((void **)&input, size * sizeof(float));

    cudaMemcpy(input, cpu_input, size * sizeof(float), cudaMemcpyHostToDevice);
    int *inputShape;
    cudaMalloc((void **)&inputShape, nDims * sizeof(float));
    cudaMemcpy(inputShape, cpu_inputShape, nDims * sizeof(float), cudaMemcpyHostToDevice);
    int stride = 1, temp = 1; // stride=[JKS, KS, S, 1][axis]

    for (int i = nDims - 1; i >= 0; --i)
    { // must i = nDims - 1, --i; can't i = 0, i++
        if (i == axis)
        {
            stride = temp;
        }
        temp *= cpu_inputShape[i];
    }

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    if (dimsize > 1024)
    {
        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<1024>
            <<<num_blocks, BLOCK_DIM>>>(input, input, size, dimsize, stride);
    }
    else
    {

        int num_block_x = (num_blocks + warpSize - 1) / warpSize;
        dim3 block_dim(warpSize, warpSize, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<<<grid_dim, block_dim>>>(input, input, size,
                                                    dimsize, stride);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(cpu_input, input, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(inputShape);

    ela = get_walltime() - st;

    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time / 1000., ela);
}
int main()
{

    int nDims = 2;
    int cpu_inputShape[nDims] = {2, 5};
    int axis = 1;

    int size = 1;
    for (int i = nDims - 1; i >= 0; --i)
    {
        size *= cpu_inputShape[i];
    }
    // printf("stride:%d\n",stride);

    float cpu_input[size] = {0, 1, 2, 3, 4, 10000, 10001, 10002, 10003, 10004};
    softmax(cpu_input, size, cpu_inputShape, axis, nDims);

    float s = 0;
    for (int i = 0; i < size; i++)
    {
        s += cpu_input[i];
        printf("softmax:%.4e\n", cpu_input[i]);
    }
    printf("s:%.3e\n", s);

    return 0;
}


