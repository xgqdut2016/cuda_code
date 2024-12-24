#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
template <int BLOCK_DIM>
__global__ void softmaxKernel(float *input, float *output, int size)
{

    __shared__ float maxData[BLOCK_DIM];
    maxData[threadIdx.x] = -__FLT_MAX__;
    for (int i = threadIdx.x; i < size; i += BLOCK_DIM)
    {
        maxData[threadIdx.x] = max(maxData[threadIdx.x], input[i]);
    }
    for (int strip = blockDim.x / 2; strip > 0; strip /= 2)
    {
        if (threadIdx.x < strip)
        {
            maxData[threadIdx.x] = max(maxData[threadIdx.x], maxData[threadIdx.x + strip]);
        }
        __syncthreads();
    }
    __syncthreads();
    __shared__ float sumData[BLOCK_DIM];
    sumData[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < size; i += BLOCK_DIM)
    {
        sumData[threadIdx.x] += __expf(input[i] - maxData[0]);
    }
    for (int strip = blockDim.x / 2; strip > 0; strip /= 2)
    {
        if (threadIdx.x < strip)
        {
            sumData[threadIdx.x] += sumData[threadIdx.x + strip];
        }
        __syncthreads();
    }
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        output[index] = __expf(input[index] - maxData[0]) * __fdividef(1.0F, sumData[0]);
    }
}
void cpuSoftmax(float *cpu_input, float *cpu_output, int size)
{
    double st, ela;
    st = get_walltime();

    float *input, *output;
    cudaMalloc((void **)&input, size * sizeof(float));
    cudaMalloc((void **)&output, size * sizeof(float));

    cudaMemcpy(input, cpu_input, size * sizeof(float), cudaMemcpyHostToDevice);
    int BLOCK_DIM = 1024;
    int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 block_dim(BLOCK_DIM, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    softmaxKernel<1024><<<grid_dim, block_dim>>>(input, output, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(cpu_output, output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(output);

    ela = get_walltime() - st;

    printf("kernel time:%.4f, use time:%.4f\n", ker_time / 1000., ela);
}

int main()
{
    float *cpu_input, *cpu_output;
    int size = 1600;

    cpu_input = (float *)malloc(size * sizeof(float));
    cpu_output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        cpu_input[i] = i % 3;
    }
    cpuSoftmax(cpu_input, cpu_output, size);
    free(cpu_input);
    free(cpu_output);
    return 0;
}


