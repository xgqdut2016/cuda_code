#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>
void getThreadNum();
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

template <int Br, int Bc, int Rq>
__device__ void matmulRQK(const float *__restrict inputQ,
                          const float *__restrict inputK, float *Qds,
                          float *Kds, int N, int d, int width, int indQ,
                          int indK, float *regLeft, float *val)
{

    for (int ph = 0; ph < width; ph++)
    {
        if (threadIdx.y < Bc)
        {
            Kds[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        if (threadIdx.y < Bc)
        {
            if (indK < N && threadIdx.y + ph * Bc < d)
            {
                Kds[threadIdx.y * Bc + threadIdx.x] =
                    inputK[indK * d + threadIdx.y + ph * Bc];
            }
        }

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + index_q < N && threadIdx.x + ph * Bc < d)
            {
                Qds[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] =
                    inputQ[(indQ + index_q) * d + threadIdx.x + ph * Bc];
            }
            else
            {
                Qds[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();                         // 读取(Bc+Bc),计算1
        for (int index = 0; index < Bc; index++) // 一个线程读取(Rq*Bc+Bc),计算Rq/（Rq+1)/Bc
        {
            for (int index_q = 0; index_q < Rq; index_q++) //[Br*Rq,Bc] @[Bc,Bc]
            {
                regLeft[index_q] =
                    Qds[(threadIdx.y * Rq + index_q) * Bc + index]; // Bc*Rq,Rq
                val[index_q] =
                    std::fma(regLeft[index_q], Kds[index * Bc + threadIdx.x],
                             val[index_q]);
            }
        }
        __syncthreads();
    }
}
template <int Br, int Bc, int Rq, int Rv>
__device__ void matmulSV(float *sumQK, const float *__restrict inputV,
                         float *Vds, int N, int d, int j, int indQ, int indK,
                         int indV, float *regLeft, float *regRight, float *val,
                         float *newMax, float *sumSV)
{

    if (threadIdx.y < Bc)
    {
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            if (threadIdx.y + j * Bc < N && indV + index_v < d)
            {
                Vds[threadIdx.y * Bc * Rv + threadIdx.x * Rv + index_v] =
                    inputV[(threadIdx.y + j * Bc) * d + indV + index_v];
            }
            else
            {
                Vds[threadIdx.y * Bc * Rv + threadIdx.x * Rv + index_v] = 0.0f;
            }
        }
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {
        if (indQ + index_q < N && indK < N)
        {
            sumQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] =
                __expf(val[index_q] - newMax[index_q]);
        }
        else
        {

            sumQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();
    for (int phc = 0; phc < Bc; phc++)
    {
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            regLeft[index_q] = sumQK[(threadIdx.y * Rq + index_q) * Bc + phc];
        }
        //--------
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            regRight[index_v] = Vds[phc * Bc * Rv + threadIdx.x * Rv + index_v];
        }
        //--------
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                sumSV[index_q * Rv + index_v] +=
                    regLeft[index_q] * regRight[index_v];
            }
        }
        // for (int index_q = 0; index_q < Rq; index_q++) {
        //     for (int index_v = 0; index_v < Rv; index_v++) {
        //         sumSV[index_q * Rv + index_v] +=
        //             sumQK[(threadIdx.y * Rq + index_q) * Bc + phc] *
        //             Vds[phc * Bc * Rv + threadIdx.x * Rv + index_v];
        //     }
        // }
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
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}

template <int Br, int Bc, int Rq, int Rv>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output)
{

    __shared__ float sumQK[Rq * Br * Bc];
    float sumSV[Rq * Rv];
    __shared__ float block_max[Rq][Br];
    __shared__ float block_sum[Rq][Br];

    int indV = Rv * (threadIdx.x + blockIdx.x * blockDim.x);
    int indQ = Rq * (threadIdx.y + blockIdx.y * blockDim.y);
    float newMax[Rq];
    float oldMax[Rq];
    float newSum[Rq];

    float out[Rq * Rv];

    for (int index_q = 0; index_q < Rq; index_q++)
    {
        newMax[index_q] = -__FLT_MAX__;
        oldMax[index_q] = -__FLT_MAX__;
        newSum[index_q] = 0.0f;
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            out[index_q * Rv + index_v] = 0.0f;
        }
    }

    float regTmp[Rq];
    int Tc = (N + Bc - 1) / Bc;
    __shared__ float Vds[Bc * Bc * Rv];
    __shared__ float Qds[Rq * Br * Bc];
    __shared__ float Kds[Bc * Bc];
    float regLeft[Rq];
    float regRight[Rv];
    float val[Rq];
    int width = (d + Bc - 1) / Bc;
    for (int j = 0; j < Tc; j++)
    {

        int indK = threadIdx.x + j * Bc;
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            val[index_q] = 0.0f;
        }

        matmulRQK<Br, Bc, Rq>(inputQ, inputK, Qds, Kds, N, d, width, indQ, indK,
                              regLeft, val);
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + index_q < N && indK < N)
            {

                regTmp[index_q] = val[index_q];
            }
            else
            {

                regTmp[index_q] = -__FLT_MAX__;
            }
        }
        __syncthreads();
        // softmax reduce
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            regTmp[index_q] = WarpAllReduce<MaxOp, float, Bc>(regTmp[index_q]);
            if (threadIdx.x == 0)
            {
                block_max[index_q][threadIdx.y] = regTmp[index_q];
            }
        }
        __syncthreads();
        //--------------------
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + index_q < N && indK < N)
            {
                regTmp[index_q] =
                    __expf(val[index_q] - block_max[index_q][threadIdx.y]);
            }
            else
            {

                regTmp[index_q] = 0.0f;
            }
        }
        __syncthreads();
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            regTmp[index_q] = WarpAllReduce<SumOp, float, Bc>(regTmp[index_q]);
            if (threadIdx.x == 0)
            {
                block_sum[index_q][threadIdx.y] = regTmp[index_q];
            }
        }
        __syncthreads();
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (newMax[index_q] > block_max[index_q][threadIdx.y])
            {
                newSum[index_q] = std::fma(
                    block_sum[index_q][threadIdx.y],
                    __expf(block_max[index_q][threadIdx.y] - newMax[index_q]),
                    newSum[index_q]);
            }
            else
            {
                newSum[index_q] = std::fma(
                    newSum[index_q],
                    __expf(newMax[index_q] - block_max[index_q][threadIdx.y]),
                    block_sum[index_q][threadIdx.y]);

                newMax[index_q] = block_max[index_q][threadIdx.y];
            }
        }
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                sumSV[index_q * Rv + index_v] = 0.0f;
            }
        }
        matmulSV<Br, Bc, Rq, Rv>(sumQK, inputV, Vds, N, d, j, indQ, indK, indV,
                                 regLeft, regRight, val, newMax, sumSV);
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                out[index_q * Rv + index_v] = std::fma(
                    __expf(oldMax[index_q] - newMax[index_q]),
                    out[index_q * Rv + index_v], sumSV[index_q * Rv + index_v]);
            }
        }

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            oldMax[index_q] = newMax[index_q];
        }

        //__syncthreads();
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {
        float inv = __fdividef(1.0F, newSum[index_q]);
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            if (indQ + index_q < N && indV + index_v < d)
            {
                output[(indQ + index_q) * d + indV + index_v] =
                    out[index_q * Rv + index_v] * inv;
            }
        }
    }
}
void attention(float *cpu_Q, float *cpu_K, float *cpu_V, int N, int d, float *cpu_output)
{
    double st, ela;
    st = get_walltime();

    float *inputQ, *inputK, *inputV, *output;
    cudaMalloc((void **)&inputQ, N * d * sizeof(float));
    cudaMalloc((void **)&inputK, N * d * sizeof(float));
    cudaMalloc((void **)&inputV, N * d * sizeof(float));

    cudaMalloc((void **)&output, N * d * sizeof(float));
    cudaMemcpy(inputQ, cpu_Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputK, cpu_K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputV, cpu_V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    int Br = 32;
    int Bc = 32; // Br>=Bc
    int Rq = 2;
    int Rv = 4;
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);
    _attentionKernel<32, 32, 2, 4>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        _attentionKernel<32, 32, 2, 4><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(cpu_output, output, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inputQ);
    cudaFree(inputK);
    cudaFree(inputV);

    cudaFree(output);

    ela = get_walltime() - st;

    printf("[N, d]:[%d, %d]\n", N, d);
    printf("GPU time:%.4f\n", ela);
    printf("kernel time:%.4f s, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    // getThreadNum();
}
int main()
{
    int N = 1024;
    int d = 1024;

    int size = N * d;

    float *cpu_Q, *cpu_K, *cpu_V, *cpu_output;
    cpu_Q = (float *)malloc(size * sizeof(float));
    cpu_K = (float *)malloc(size * sizeof(float));
    cpu_V = (float *)malloc(size * sizeof(float));
    cpu_output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        cpu_Q[i] = i % 4;
        cpu_K[i] = i % 4;
        cpu_V[i] = i % 4;
        // printf("Q:%.4f\n",cpu_Q[i]);
    }

    attention(cpu_Q, cpu_K, cpu_V, N, d, cpu_output);
    // for (int i = 0; i < 10; i++)
    // {

    //     printf("out:%.6e\n", cpu_output[i]);
    // }

    free(cpu_Q);
    free(cpu_K);
    free(cpu_V);
    free(cpu_output);

    return 0;
}
void getThreadNum()
{
    int deviceCount;

    cudaGetDeviceCount(&deviceCount); // Returns in *deviceCount the number of devices
    printf("deviceCount: %d\n ", deviceCount);

    if (deviceCount == 0)
    {
        printf("error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    cudaSetDevice(dev); // Sets dev=0 device as the current device for the calling host thread.

    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);
    printf("name:%s\n", devProps.name);
    printf("totalGlobalMem: %ld\n", devProps.totalGlobalMem);
    printf("regsPerBlock: %d\n", devProps.regsPerBlock);
    printf("warpSize: %d\n", devProps.warpSize);
    printf("memPitch: %ld\n\n", devProps.memPitch);

    printf("一个线程块中可使用的最大共享内存\n");
    printf("devProps.sharedMemPerBlock: %ld Bytes \n\n", devProps.sharedMemPerBlock);

    printf("一个线程块中可包含的最大线程数量\n");
    printf("maxThreadsPerBlock: %d\n", devProps.maxThreadsPerBlock);

    printf("多维线程块数组中每一维可包含的最大线程数量\n");
    printf("maxThreadsDim[0]: %d\n", devProps.maxThreadsDim[0]);
    printf("maxThreadsDim[1]: %d\n", devProps.maxThreadsDim[1]);
    printf("maxThreadsDim[2]: %d\n\n", devProps.maxThreadsDim[2]);

    printf("一个线程格中每一维可包含的最大线程块数量\n");
    printf("maxGridSize[0]: %d\n", devProps.maxGridSize[0]);
    printf("maxGridSize[1]: %d\n", devProps.maxGridSize[1]);
    printf("maxGridSize[2]: %d\n\n", devProps.maxGridSize[2]);

    printf("clockRate: %d\n", devProps.clockRate);
    printf("totalConstMem: %ld\n", devProps.totalConstMem);
    printf("textureAlignment: %ld\n\n", devProps.textureAlignment);

    printf("计算能力：%d.%d\n", devProps.major, devProps.minor);
}
