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

template <int Br, int Bc>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,

                                 float *__restrict output)
{

    //  一个线程块处理Q的Br行，V的Bc列，以及全部的K,blockDim.x=Bc,blockDim.y=Br
    int Tc = (N + Bc - 1) / Bc; // 遍历矩阵inputK的N行需要的循环次数

    __shared__ float sumQK[Br * Bc];
    __shared__ float sumSV[Br * Bc];
    __shared__ float block_max[Br * Bc];
    __shared__ float block_sum[Br * Bc];
    __shared__ float Vds[Bc * Bc];
    __shared__ float Qds[Br * Bc];
    __shared__ float Kds[Bc * Bc];
    int indV = threadIdx.x +
               blockIdx.x * blockDim.x; // 对应的是当前block需要处理的V的列索引
    int indQ = threadIdx.y + blockIdx.y * blockDim.y;
    float newMax;
    float oldMax;
    float newSum;
    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 1.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++)
    {
        sumSV[threadIdx.y * Bc + threadIdx.x] =
            0.0f;                        // 每次循环需要重新初始化为0
        int indK = threadIdx.x + j * Bc; // 通过j循环来遍历K的行索引
        float sum_qk = 0.0f;
        for (int ph = 0; ph < gridDim.x; ph++)
        {
            if (indQ < N && threadIdx.x + ph * Bc < d)
            {
                Qds[threadIdx.y * Bc + threadIdx.x] =
                    inputQ[indQ * d + threadIdx.x + ph * Bc];
            }
            else
            {
                Qds[threadIdx.y * Bc + threadIdx.x] = 0.0f;
            }
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

            __syncthreads();
            for (int index = 0; index < Bc; index++)
            {
                sum_qk = std::fma(Qds[threadIdx.y * Bc + index],
                                  Kds[index * Bc + threadIdx.x], sum_qk);
                // if (index + ph * Bc < d) {
                //     sum_qk += Qds[threadIdx.y * Bc + index] *
                //               Kds[index * Bc + threadIdx.x];
                // }
                // if (index + ph * Bc < d) {
                //     sum_qk =
                //         std::fma(inputQ[indQ * d + index + ph * Bc],
                //                  inputK[indK * d + index + ph * Bc], sum_qk);
                // }
            }
            __syncthreads();
        }

        if (indQ < N && indK < N)
        {
            block_max[threadIdx.y * Bc + threadIdx.x] = sum_qk;
            block_sum[threadIdx.y * Bc + threadIdx.x] = 1.0f;
            sumQK[threadIdx.y * Bc + threadIdx.x] =
                sum_qk; // 存储QK的结果，循环内部不做修改
        }
        else
        {
            block_max[threadIdx.y * Bc + threadIdx.x] = -__FLT_MAX__;
            block_sum[threadIdx.y * Bc + threadIdx.x] = 0.0f;
            sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int strip = Bc / 2; strip > 0; strip /= 2)
        {
            if (threadIdx.x < strip)
            {
                if (block_max[threadIdx.y * Bc + threadIdx.x] >
                    block_max[threadIdx.y * Bc + threadIdx.x + strip])
                {
                    block_sum[threadIdx.y * Bc + threadIdx.x] =
                        block_sum[threadIdx.y * Bc + threadIdx.x] +
                        block_sum[threadIdx.y * Bc + threadIdx.x + strip] *
                            __expf(block_max[threadIdx.y * Bc + threadIdx.x +
                                             strip] -
                                   block_max[threadIdx.y * Bc + threadIdx.x]);
                }
                else
                {
                    block_sum[threadIdx.y * Bc + threadIdx.x] =
                        block_sum[threadIdx.y * Bc + threadIdx.x + strip] +
                        block_sum[threadIdx.y * Bc + threadIdx.x] *
                            __expf(block_max[threadIdx.y * Bc + threadIdx.x] -
                                   block_max[threadIdx.y * Bc + threadIdx.x +
                                             strip]);
                    block_max[threadIdx.y * Bc + threadIdx.x] =
                        block_max[threadIdx.y * Bc + threadIdx.x + strip];
                }
            }
            __syncthreads();
        }

        if (newMax >
            block_max[threadIdx.y *
                      Bc]) // threadIdx.y=0存储的是对应分块矩阵的局部max
        {                  // 为了获得全局max，需要不断更新newMax和threadIdx.y=0的比较结果
            newSum = newSum + block_sum[threadIdx.y * Bc] *
                                  __expf(block_max[threadIdx.y * Bc] - newMax);
        }
        else
        {
            newSum = block_sum[threadIdx.y * Bc] +
                     newSum * __expf(newMax - block_max[threadIdx.y * Bc]);
            newMax = block_max[threadIdx.y * Bc];
        }
        if (threadIdx.y < Bc)
        { // threadIdx.y的范围必须>=Bc
            if (threadIdx.y + j * Bc < N && indV < d)
            {
                Vds[threadIdx.x * Bc + threadIdx.y] =
                    inputV[(threadIdx.y + j * Bc) * d + indV];
            }
            else
            {
                Vds[threadIdx.x * Bc + threadIdx.y] = 0.0f;
            }
        }
        if (indQ < N && indK < N)
        {
            sumQK[threadIdx.y * Bc + threadIdx.x] =
                __expf(sumQK[threadIdx.y * Bc + threadIdx.x] - newMax);
        }
        else
        {
            sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int phc = 0; phc < Bc; phc++) // 这里开始做最后和V的matmul
        {
            sumSV[threadIdx.y * Bc + threadIdx.x] = std::fma(
                sumQK[threadIdx.y * Bc + phc], Vds[threadIdx.x * Bc + phc],
                sumSV[threadIdx.y * Bc + threadIdx.x]);
        }
        out = __expf(oldMax - newMax) * out +
              sumSV[threadIdx.y * Bc + threadIdx.x];
        oldMax = newMax;

        __syncthreads();
    }
    if (indQ < N && indV < d)
    {
        output[indQ * d + indV] = out * __fdividef(1.0F, newSum);
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

    int num_block_x = (d + Bc - 1) / Bc;
    int num_block_y = (N + Br - 1) / Br;
    dim3 block_dim(Bc, Br, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    _attentionKernel<32, 32>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        _attentionKernel<32, 32><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
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
