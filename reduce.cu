#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <cub/block/block_reduce.cuh>
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
float addCpu(float *hostA, int n)
{
    float tmp = 0.0f; // 表示C++中的负无穷
    for (int i = 0; i < n; i++)
    {
        tmp += hostA[i];
    }
    return tmp;
}
template <int BLOCK_DIM>
__global__ void addKernel(float *dA, int n, float *globalMax, int strategy)
{
    __shared__ float tmpSum[BLOCK_DIM];
    float tmp = 0.0f;
    for (int id = threadIdx.x; id < n; id += BLOCK_DIM)
    {
        tmp += dA[id];
    }
    tmpSum[threadIdx.x] = tmp;
    if (strategy == 0)
    {
        for (int step = 1; step < BLOCK_DIM; step *= 2)
        {
            if (threadIdx.x % (2 * step) == 0)
            {
                tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            globalMax[0] = tmpSum[0];
        }
    }
    else if (strategy == 1)
    {
        for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
        {
            if (threadIdx.x < step)
            {
                tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            globalMax[0] = tmpSum[0];
        }
    }
    else if (strategy == 2)
    {
        typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce; //<float,..>里面的float表示返回值的类型
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float block_sum = BlockReduce(temp_storage).Reduce(tmpSum[threadIdx.x], cub::Sum());
        if (threadIdx.x == 0)
        {
            globalMax[0] = block_sum;
        }
    }
    else
    {
        __shared__ float val[32];
        float data = tmpSum[threadIdx.x];
        data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
        data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
        data += __shfl_down_sync(0xffffffff, data, 4);
        data += __shfl_down_sync(0xffffffff, data, 2);
        data += __shfl_down_sync(0xffffffff, data, 1);
        if (threadIdx.x % 32 == 0)
        {
            val[threadIdx.x / 32] = data;
        }
        __syncthreads();
        if (threadIdx.x < 32)
        {
            data = val[threadIdx.x];
            data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
            data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
            data += __shfl_down_sync(0xffffffff, data, 4);
            data += __shfl_down_sync(0xffffffff, data, 2);
            data += __shfl_down_sync(0xffffffff, data, 1);
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            globalMax[0] = data;
        }
    }
}
int main()
{
    float *hostA;
    int n = 102400;
    int strategy = 2;
    hostA = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        hostA[i] = (i % 10) * 1e-1;
    }
    float hostMax;
    double st, ela;
    st = get_walltime();

    float *dA, *globalMax;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&globalMax, sizeof(float));
    cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int BLOCK_DIM = 1024;
    int num_block_x = n / BLOCK_DIM;
    int num_block_y = 1;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    addKernel<1024><<<grid_dim, block_dim>>>(dA, n, globalMax, strategy);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(&hostMax, globalMax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(globalMax);
    ela = get_walltime() - st;
    printf("n = %d: GPU use time:%.4f, kernel time:%.4f\n", n, ela, ker_time / 1000.0);
    printf("CPU sum:%.2f, GPU sum:%.2f\n", addCpu(hostA, n), hostMax);
    free(hostA);

    return 0;
}
