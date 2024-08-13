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

const int Rq = 8;
const int Rv = 8; // 必须是4的倍数
const int Br = 16;
const int Bc = 16;
const int Bk = 8; // 必须是4的倍数
const int Bd = 8;
const int numQ = Rq * Br;
const int numK = Bk * Bc;
const int numV = Rv * Bc;

__device__ void matmulRQK(const float *__restrict inputQ,
                          const float *__restrict inputK, float *shareQK,
                          float *shareVK, int N, int d, int width, int indQ,
                          int indK, float *val)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float4 a[1];
    float4 b[1];
    float com_a[Rq];
    float com_b[Bk];

    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_n = tid / 128;
    int smem_b_k = tid % 128;
    // float tmp[64];
    // memset(tmp, 0.0f, sizeof(tmp));
    int ph = 0;
    (float4 &)a[0] =
        (float4 &)inputQ[(indQ + smem_a_m) * d + ph * Bd + 4 * smem_a_k];

    (float4 &)b[0] =
        (float4 &)inputK[(indK + smem_b_k) * d + Bd * ph + 4 * smem_b_n];
    shareQK[(4 * smem_a_k) * numQ + smem_a_m] = a[0].x;
    shareQK[(4 * smem_a_k + 1) * numQ + smem_a_m] = a[0].y;
    shareQK[(4 * smem_a_k + 2) * numQ + smem_a_m] = a[0].z;
    shareQK[(4 * smem_a_k + 3) * numQ + smem_a_m] = a[0].w;

    shareVK[(4 * smem_b_n) * numK + smem_b_k] = b[0].x;
    shareVK[(4 * smem_b_n + 1) * numK + smem_b_k] = b[0].y;
    shareVK[(4 * smem_b_n + 2) * numK + smem_b_k] = b[0].z;
    shareVK[(4 * smem_b_n + 3) * numK + smem_b_k] = b[0].w;
    __syncthreads();
    for (int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] =
            (float4 &)inputQ[(indQ + smem_a_m) * d + ph * Bd + 4 * smem_a_k];

        (float4 &)b[0] =
            (float4 &)inputK[(indK + smem_b_k) * d + Bd * ph + 4 * smem_b_n];

        for (int index = 0; index < Bd; index++)
        {
            (float4 &)com_a[0] =
                (float4 &)shareQK[index * numQ + threadIdx.y * Rq +
                                  (ph - 1) % 2 * numQ * Bd];
            (float4 &)com_a[4] =
                (float4 &)shareQK[index * numQ + threadIdx.y * Rq + 4 +
                                  (ph - 1) % 2 * numQ * Bd];
            (float4 &)com_b[0] =
                (float4 &)shareVK[index * numK + threadIdx.x * Bk +
                                  (ph - 1) % 2 * numK * Bd];
            (float4 &)com_b[4] =
                (float4 &)shareVK[index * numK + threadIdx.x * Bk + 4 +
                                  (ph - 1) % 2 * numK * Bd];

            for (int index_q = 0; index_q < Rq; index_q++)
            {
                for (int index_k = 0; index_k < Bk; index_k++)
                {

                    val[index_q * Rq + index_k] += com_a[index_q] * com_b[index_k];
                }
            }
        }

        shareQK[(4 * smem_a_k) * numQ + smem_a_m + (ph % 2) * numQ * Bd] =
            a[0].x;
        shareQK[(4 * smem_a_k + 1) * numQ + smem_a_m + (ph % 2) * numQ * Bd] =
            a[0].y;
        shareQK[(4 * smem_a_k + 2) * numQ + smem_a_m + (ph % 2) * numQ * Bd] =
            a[0].z;
        shareQK[(4 * smem_a_k + 3) * numQ + smem_a_m + (ph % 2) * numQ * Bd] =
            a[0].w;

        shareVK[(4 * smem_b_n) * numK + smem_b_k + (ph % 2) * numK * Bd] =
            b[0].x;
        shareVK[(4 * smem_b_n + 1) * numK + smem_b_k + (ph % 2) * numK * Bd] =
            b[0].y;
        shareVK[(4 * smem_b_n + 2) * numK + smem_b_k + (ph % 2) * numK * Bd] =
            b[0].z;
        shareVK[(4 * smem_b_n + 3) * numK + smem_b_k + (ph % 2) * numK * Bd] =
            b[0].w;

        __syncthreads();
    }
    ph = width;
    for (int index = 0; index < Bd; index++)
    {
        (float4 &)com_a[0] =
            (float4 &)shareQK[index * numQ + threadIdx.y * Rq +
                              (ph - 1) % 2 * numQ * Bd];
        (float4 &)com_a[4] =
            (float4 &)shareQK[index * numQ + threadIdx.y * Rq + 4 +
                              (ph - 1) % 2 * numQ * Bd];
        (float4 &)com_b[0] =
            (float4 &)shareVK[index * numK + threadIdx.x * Bk +
                              (ph - 1) % 2 * numK * Bd];
        (float4 &)com_b[4] =
            (float4 &)shareVK[index * numK + threadIdx.x * Bk + 4 +
                              (ph - 1) % 2 * numK * Bd];

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_k = 0; index_k < Bk; index_k++)
            {

                val[index_q * Rq + index_k] += com_a[index_q] * com_b[index_k];
            }
        }
    }
}

__device__ void matmulSV(float *shareQK, const float *__restrict inputV,
                         float *shareVK, int N, int d, int j, int indQ,
                         int indK, int indV, float *val, float *newMax,
                         float *sumSV)
{
    for (int index_k = 0; index_k < Bk; index_k++)
    {
        for (int id = 0; id < Rv; id += 4)
        {
            (float4 &)shareVK[threadIdx.y * numV + threadIdx.x * Rv + id] =
                (float4 &)inputV[(indK + threadIdx.y * Bk + index_k) * d +
                                 indV + threadIdx.x * Rv + id];
        }
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            if (indK + threadIdx.y * Bk + index_k >= N ||
                indV + threadIdx.x * Rv + index_v >= d)
            {
                shareVK[threadIdx.y * numV + threadIdx.x * Rv + index_v] = 0.0f;
            }
        }
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + threadIdx.y * Rq + index_q < N &&
                indK + Bk * threadIdx.x + index_k < N)
            {
                shareQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] =
                    __expf(val[index_q * Bk + index_k] - newMax[index_q]);
            }
            else
            {

                shareQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        for (int phc = 0; phc < Bc; phc++)
        {
            for (int index_q = 0; index_q < Rq; index_q++)
            {
                for (int index_v = 0; index_v < Rv; index_v++)
                {
                    sumSV[index_q * Rv + index_v] +=
                        shareQK[(threadIdx.y * Rq + index_q) * Bc + phc] *
                        shareVK[phc * numV + threadIdx.x * Rv + index_v];
                }
            }
        }
        __syncthreads();
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
          int thread_group_width = 32>
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

    __shared__ float shareQK[numQ * Bc];
    __shared__ float shareVK[Bc * numV];
    __shared__ float block_max[numQ];
    __shared__ float block_sum[numQ];
    float sumSV[Rq * Rv] = {0.0f};
    float newMax[Rq];
    float oldMax[Rq];
    float newSum[Rq] = {0.0f};

    float val[Rq * Bk];

    int indV = Rv * blockIdx.x * blockDim.x;
    int indQ = Rq * blockIdx.y * blockDim.y;

    for (int index_q = 0; index_q < Rq; index_q++)
    {
        newMax[index_q] = -__FLT_MAX__;
        oldMax[index_q] = -__FLT_MAX__;
    }

    int Tc = (N + numK - 1) / (numK);

    int width = (d + Bd - 1) / Bd;
    for (int j = 0; j < Tc; j++)
    {

        int indK = j * numK;
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_k = 0; index_k < Bk; index_k++)
            {

                val[index_q * Bk + index_k] = 0.0f;
            }
        }
        matmulRQK(inputQ, inputK, shareQK, shareVK, N, d, width, indQ, indK,
                  val);
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            float tmpReduceMax = -__FLT_MAX__;
            for (int index_k = 0; index_k < Bk; index_k++)
            {
                if (indQ + threadIdx.y * Rq + index_q < N &&
                    indK + Bk * threadIdx.x + index_k < N)
                {

                    tmpReduceMax =
                        max(tmpReduceMax, val[index_q * Bk + index_k]);
                }
            }
            __syncthreads();
            tmpReduceMax = WarpAllReduce<MaxOp, float, Bc>(tmpReduceMax);
            if (threadIdx.x == 0)
            {
                block_max[threadIdx.y * Rq + index_q] = tmpReduceMax;
            }
            __syncthreads();
            float tmpReduceSum = 0.0f;
            for (int index_k = 0; index_k < Bk; index_k++)
            {
                if (indQ + threadIdx.y * Rq + index_q < N &&
                    indK + Bk * threadIdx.x + index_k < N)
                {
                    tmpReduceSum +=
                        __expf(val[index_q * Bk + index_k] -
                               block_max[threadIdx.y * Rq + index_q]);
                }
            }
            __syncthreads();
            tmpReduceSum = WarpAllReduce<SumOp, float, Bc>(tmpReduceSum);
            if (threadIdx.x == 0)
            {
                block_sum[threadIdx.y * Rq + index_q] = tmpReduceSum;
            }
            __syncthreads();
            if (newMax[index_q] > block_max[threadIdx.y * Rq + index_q])
            {
                newSum[index_q] =
                    std::fma(block_sum[threadIdx.y * Rq + index_q],
                             __expf(block_max[threadIdx.y * Rq + index_q] -
                                    newMax[index_q]),
                             newSum[index_q]);
            }
            else
            {
                newSum[index_q] =
                    std::fma(newSum[index_q],
                             __expf(newMax[index_q] -
                                    block_max[threadIdx.y * Rq + index_q]),
                             block_sum[threadIdx.y * Rq + index_q]);

                newMax[index_q] = block_max[threadIdx.y * Rq + index_q];
            }
            // PV
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                sumSV[index_q * Rv + index_v] *=
                    __expf(oldMax[index_q] - newMax[index_q]);
            }
        }

        matmulSV(shareQK, inputV, shareVK, N, d, j, indQ, indK, indV, val,
                 newMax, sumSV);

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            oldMax[index_q] = newMax[index_q];
        }

        __syncthreads();
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {
        float inv = __fdividef(1.0F, newSum[index_q]);
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            sumSV[index_q * Rv + index_v] = sumSV[index_q * Rv + index_v] * inv;
        }
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {

        for (int id = 0; id < Rv; id += 4)
        {
            if (indQ + threadIdx.y * Rq + index_q < N &&
                indV + threadIdx.x * Rv + id < d)
            {
                (float4 &)output[(indQ + threadIdx.y * Rq + index_q) * d +
                                 indV + threadIdx.x * Rv + id] =
                    (float4 &)sumSV[index_q * Rv + id];
            }
        }
    }
}
void attention(float *cpu_Q, float *cpu_K, float *cpu_V, int N, int d, float *cpu_output)
{
    cudaSetDevice(8);
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
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);

    _attentionKernel<Br, Bc, Rq, Rv>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        _attentionKernel<Br, Bc, Rq, Rv>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    }

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
