#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
const int BLOCK_DIM = 1024;
__global__ void softmax(float *input, float *output, int M, int N)
{
    int row = blockIdx.x;
    __shared__ float tmp[BLOCK_DIM];
    __shared__ float globalMax;
    __shared__ float globalSum;
    //-----------
    float val = -__FLT_MAX__;
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        val = max(val, input[row * N + i]);
    }
    tmp[threadIdx.x] = val;
    for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalMax = tmp[0];
    }
    __syncthreads();
    //-----------

    val = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        val += __expf(input[row * N + i] - globalMax);
    }
    tmp[threadIdx.x] = val;
    for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x] += tmp[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalSum = tmp[0];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        output[row * N + i] = __expf(input[row * N + i] - globalMax) * __fdividef(1.0F, globalSum);
    }
}
void cpu_softmax(float *cpu_input, float *cpu_output, int M, int N)
{
    double st, ela;
    st = get_walltime();

    int num_block = M;
    dim3 block_dim(BLOCK_DIM, 1, 1);
    dim3 grid_dim(num_block, 1, 1);

    float *input, *output;
    cudaMalloc((void **)&input, M * N * sizeof(float));
    cudaMalloc((void **)&output, M * N * sizeof(float));
    cudaMemcpy(input, cpu_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    softmax<<<grid_dim, block_dim>>>(input, output, M, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(cpu_output, output, M * N, cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(output);

    ela = get_walltime() - st;

    printf("kernel time:%.4f, use time:%.4f\n", ker_time / 1000., ela);
}

int main()
{
    float *cpu_input, *cpu_output;
    int M = 1024;
    int N = 1024;
    cpu_input = (float *)malloc(M * N * sizeof(float));
    cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++)
    {
        cpu_input[i] = i % 10;
    }
    cpu_softmax(cpu_input, cpu_output, M, N);
    for (int i = 0; i < 10; i++)
    {
        printf("%.4e ", cpu_output[i]);
    }
    printf("\n");
    free(cpu_input);
    free(cpu_output);
    return 0;
}
