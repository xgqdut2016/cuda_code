#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;
const int warpSize = 32;
__global__ void row_wmma_ker(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int lda = K; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = N;
    int ldc = N;

    int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpY = (blockIdx.y * blockDim.y + threadIdx.y);
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int aRow = warpX * WMMA_M; //(aRow, aCol),x方向数多少个WMMA_M
    int bCol = warpY * WMMA_N;
    for (int i = 0; i < K; i += WMMA_K)
    {
        int aCol = i;
        int bRow = i;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // 读取A,B矩阵里面子矩阵的元素
            wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);
            // 子矩阵做乘法
            wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        }
    }
    int cRow = warpX * WMMA_M;
    int cCol = warpY * WMMA_N;
    if (cRow < M && cCol < N)
    {
        // Store the output
        wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}

double
get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    float tmp = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    return error;
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_DIM_x = 32; // 必须取32，否则结果报错
    int BLOCK_DIM_y = 4;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim((M + (WMMA_M * BLOCK_DIM_x / warpSize - 1)) / (WMMA_M * BLOCK_DIM_x / warpSize), (N + WMMA_N * BLOCK_DIM_y - 1) / (WMMA_N * BLOCK_DIM_y), 1);
    float ker_time = 0;
    row_wmma_ker<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);

    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        row_wmma_ker<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
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

    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
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
    float error = compare(hostC, serialC, M, N);
    printf("CPU time:%.2f, error:%.4e\n", ela, error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}
