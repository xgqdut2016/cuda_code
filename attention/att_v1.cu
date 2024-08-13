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

__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d, int Br, int Bc,
                                 float *__restrict output)
{
    // 一个线程块处理Q的Br行，V的Bc列，以及全部的K,blockDim.x=Br,blockDim.y=Bc
    int Tc = (N + Bc - 1) / Bc;                       // 遍历矩阵inputK的N行需要的循环次数
    extern __shared__ float sram[];                   // 必须要有extern
    float *block_sum = sram;                          // 形状为[Br,Bc]，为后面softmax计算sum做准备
    float *block_max = sram + Br * Bc;                // 形状为[Br,Bc]，为后面softmax计算max做准备
    float *sumQK = sram + Br * Bc * 2;                // 形状为[Br,Bc]，存储的是QK.T的结果
    float *sumSV = sram + Br * Bc * 3;                // 形状为[Br,Bc]，存储的是softmax(QK.T)V的结果
    int indQ = threadIdx.x + blockIdx.x * blockDim.x; // 对应的是当前block需要处理的Q的行索引
    int indV = threadIdx.y + blockIdx.y * blockDim.y; // 对应的是当前block需要处理的V的列索引
    float newMax;                                     // newMax就是算法里面的m_{i,j}
    float oldMax;
    float newSum; // newSum就是l_{i,j}

    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 0.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++)
    {
        sumSV[threadIdx.x * Bc + threadIdx.y] = 0.0f; // 每次循环需要重新初始化为0
        int indK = threadIdx.y + j * Bc;              // 通过j循环来遍历K的行索引
        float sum_qk = 0.0f;
        for (int index = 0; index < d; index++)
        {
            sum_qk += inputQ[indQ * d + index] * inputK[indK * d + index];
        }
        if (indQ < N && indK < N)
        {

            block_max[threadIdx.x * Bc + threadIdx.y] = sum_qk; // 后面针对threadIdx.y做规约会修改元素内容
            sumQK[threadIdx.x * Bc + threadIdx.y] = sum_qk;     // 存储QK的结果，循环内部不做修改
            block_sum[threadIdx.x * Bc + threadIdx.y] = 1.0f;
        }
        else
        {
            sumQK[threadIdx.x * Bc + threadIdx.y] = 0.0f;
            block_max[threadIdx.x * Bc + threadIdx.y] = -__FLT_MAX__;
            block_sum[threadIdx.x * Bc + threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int strip = Bc / 2; strip > 0; strip /= 2) // 这部分规约可以理解为二维block的softmax规约，一边算max，一边算sum
        {
            if (threadIdx.y < strip)
            {
                if (block_max[threadIdx.x * Bc + threadIdx.y] >
                    block_max[threadIdx.x * Bc + threadIdx.y + strip])
                {
                    block_sum[threadIdx.x * Bc + threadIdx.y] =
                        block_sum[threadIdx.x * Bc + threadIdx.y] +
                        block_sum[threadIdx.x * Bc + threadIdx.y + strip] *
                            __expf(block_max[threadIdx.x * Bc + threadIdx.y + strip] -
                                   block_max[threadIdx.x * Bc + threadIdx.y]);
                }
                else
                {
                    block_sum[threadIdx.x * Bc + threadIdx.y] =
                        block_sum[threadIdx.x * Bc + threadIdx.y + strip] +
                        block_sum[threadIdx.x * Bc + threadIdx.y] *
                            __expf(block_max[threadIdx.x * Bc + threadIdx.y] -
                                   block_max[threadIdx.x * Bc + threadIdx.y + strip]);
                    block_max[threadIdx.x * Bc + threadIdx.y] =
                        block_max[threadIdx.x * Bc + threadIdx.y + strip];
                }
            }
            __syncthreads();
        } // 规约结果存储在threadIdx.y=0的位置
        if (newMax > block_max[threadIdx.x * Bc]) // threadIdx.y=0存储的是对应分块矩阵的局部max
        {                                         // 为了获得全局max，需要不断更新newMax和threadIdx.y=0的比较结果
            newSum = newSum + block_sum[threadIdx.x * Bc] *
                                  __expf(block_max[threadIdx.x * Bc] - newMax);
        }
        else
        {
            newSum = block_sum[threadIdx.x * Bc] +
                     newSum * __expf(newMax - block_max[threadIdx.x * Bc]);
            newMax = block_max[threadIdx.x * Bc];
        }

        __syncthreads();
        for (int phc = 0; phc < Bc; phc++) // 这里开始做最后和V的matmul
        {
            if (phc + j * Bc < N) // 注意控制范围
            {
                sumSV[threadIdx.x * Bc + threadIdx.y] += __expf(sumQK[threadIdx.x * Bc + phc] - newMax) * inputV[(phc + j * Bc) * d + indV];
            }
        }
        out = __expf(oldMax - newMax) * out + sumSV[threadIdx.x * Bc + threadIdx.y];
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
    int Bc = 32;
    int num_block_x = (N + Br - 1) / Br;
    int num_block_y = (d + Bc - 1) / Bc;
    dim3 block_dim(Br, Bc, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    int share_mem = 4 * Br * Bc * sizeof(float); // 由于global函数里面未明确分配内存，此时必须指定共享内存分配大小
    _attentionKernel<<<grid_dim, block_dim, share_mem>>>(inputQ, inputK, inputV, N, d, Br, Bc, output);
    int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        _attentionKernel<<<grid_dim, block_dim, share_mem>>>(inputQ, inputK, inputV, N, d, Br, Bc, output);
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
