#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
const int TM = 8;
const int TN = 8;
const int BLOCK_DIM_x = 16;
const int BLOCK_DIM_y = 16;
const int BM = TM * BLOCK_DIM_x;
const int BN = TN * BLOCK_DIM_y;
const int BK = 8;
double
get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
void compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    bool tmp = true;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
        if (error > 1e-5)
        {
            tmp = false;
            printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
            break;
        }
    }
    if (tmp)
    {
        printf("GPU output all right\n");
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    for (int ph = 0; ph < width; ph++)
    {
        (float4 &)SA[smem_a_m * BK + 4 * smem_a_k] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[smem_a_m * BK + 4 * smem_a_k + id] = 0.0f;
            }
            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
            }
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                int reg_c_m = threadIdx.x * TM + index_q;
                int reg_c_n = threadIdx.y * TN + index_v;
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel2nd(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    for (int ph = 0; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
            }
            else
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
            }
        }
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {

            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
            }
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                int reg_c_m = threadIdx.x * TM + index_q;
                int reg_c_n = threadIdx.y * TN + index_v;
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[index_k * BM + reg_c_m] * SB[index_k * BN + reg_c_n];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel3rd(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    float com_a[TM];
    float com_b[TN];
    for (int ph = 0; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
            }
            else
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
            }
        }
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {

            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
            }
        }

        __syncthreads();

        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM];
            (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN];
            (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4];
            for (int index_q = 0; index_q < TM; index_q++)
            {
                for (int index_v = 0; index_v < TN; index_v++)
                {
                    tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
                }
            }
        }

        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel4th(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK * 2];
    __shared__ float SB[BK * BN * 2];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    float com_a[TM];
    float com_b[TN];
    //------------
    int ph = 0;
    (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
    for (int id = 0; id < 4; id++)
    {
        if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
        {
            SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
        }
        else
        {
            SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
        }
    }
    (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
    for (int id = 0; id < 4; id++)
    {

        if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
        {

            SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
        }
    }
    __syncthreads();

    for (int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m + ph % 2 * BM * BK] = 0.0f;
            }
            else
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m + ph % 2 * BM * BK] = a[id];
            }
        }
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n + ph % 2 * BN * BK] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {

            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id + ph % 2 * BN * BK] = 0.0f;
            }
        }
        //-------------
        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
            (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
            (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
            for (int index_q = 0; index_q < TM; index_q++)
            {
                for (int index_v = 0; index_v < TN; index_v++)
                {
                    tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
                }
            }
        }
        __syncthreads();
    }
    //--------------
    ph = width;
    for (int index_k = 0; index_k < BK; index_k++)
    {
        (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
        (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
        (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
        (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
            }
        }
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixOrigin(float *dA, float *dB, float *dC, int M, int K, int N)
{

    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    float tmp[TM][TN] = {0.0f};
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                for (int s = 0; s < K; s++)
                {
                    tmp[index_q][index_v] += dA[(indA + index_q) * K + s] * dB[s * N + indB + index_v];
                }
            }
        }
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q][index_v];
            }
        }
    }
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    cudaSetDevice(3);
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixKernel3rd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    matrixKernel4th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    //   matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / 1000., ker_time);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }
    hostMatrix(hostA, hostB, hostC, M, K, N);
    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    printf("CPU time:%.2f second\n", ela);

    compare(hostC, serialC, M, N);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}
