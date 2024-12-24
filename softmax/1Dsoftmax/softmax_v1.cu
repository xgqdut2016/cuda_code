#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM 1024
#define max_function(a,b) ((a)>(b)?(a):(b))

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
__global__
void summax_soft(float *input, int size, float *res_sum, float *res_max, int reduce){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (reduce == 0){
        __shared__ float tmp_sum[(int)(BLOCK_DIM)], tmp_max[(int)(BLOCK_DIM)];
        if (i < size){
            tmp_sum[threadIdx.x] = input[i];
            tmp_max[threadIdx.x] = input[i];
        }
        else {
            tmp_sum[threadIdx.x] = 0.0f;
            tmp_max[threadIdx.x] = -__FLT_MAX__;
        }
        __syncthreads();
        for(int strip = 1; strip < blockDim.x; strip = strip*2){
            if (threadIdx.x % (2*strip) == 0){
                tmp_sum[threadIdx.x] += tmp_sum[threadIdx.x + strip];
                tmp_max[threadIdx.x] = max_function(tmp_max[threadIdx.x + strip], tmp_max[threadIdx.x]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            res_sum[blockIdx.x] = tmp_sum[0];
            res_max[blockIdx.x] = tmp_max[0];
        }
    }
    else if (reduce == 1){
        __shared__ float tmp_sum[(int)(BLOCK_DIM)], tmp_max[(int)(BLOCK_DIM)];
        if (i < size){
            tmp_sum[threadIdx.x] = input[i];
            tmp_max[threadIdx.x] = input[i];
        }
        else {
            tmp_sum[threadIdx.x] = 0.0f;
            tmp_max[threadIdx.x] = -FLT_MAX__;
        }
        __syncthreads();
        for(int strip = blockDim.x/2; strip > 0; strip = strip/2){
            if (threadIdx.x < strip){
                tmp_sum[threadIdx.x] += tmp_sum[threadIdx.x + strip];
                tmp_max[threadIdx.x] = max_function(tmp_max[threadIdx.x + strip], tmp_max[threadIdx.x]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            res_sum[blockIdx.x] = tmp_sum[0];
            res_max[blockIdx.x] = tmp_max[0];
        }
    }
    else if (reduce == 2){
        __shared__ float tmp_sum[32], tmp_max[32];
        float data = 0.0f;
        float max_data = -FLT_MAX__;
        float sum_data = 0.0f;
        if (i < size){
            max_data = input[i];
            sum_data = input[i];
        }
        for(int offset = 16; offset >0; offset/= 2){
            sum_data += __shfl_down_sync(0xffffffff, sum_data, offset);
            data = __shfl_down_sync(0xffffffff, max_data, offset);
            max_data = max_function(max_data, data);
        }

        if (threadIdx.x % 32 == 0){
            tmp_max[threadIdx.x/32] = max_data;
            tmp_sum[threadIdx.x/32] = sum_data;
        }
        __syncthreads();
        if(threadIdx.x >= 32)
            return;
        max_data = tmp_max[threadIdx.x];
        sum_data = tmp_sum[threadIdx.x];
        for(int offset = 16; offset >0; offset/= 2){
            sum_data += __shfl_down_sync(0xffffffff, sum_data, offset);
            data = __shfl_down_sync(0xffffffff, max_data, offset);
            max_data = max_function(max_data, data);
        }
        if (threadIdx.x == 0){
            res_max[blockIdx.x] = max_data;
            res_sum[blockIdx.x] = sum_data;
        }
    }
}
__global__ void softmax(float *input, float f_max, float f_sum, int size){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < size){
        input[i] = __expf(input[i] - f_max)/f_sum;
    }
}
void cpu_summax(float *cpu_input, int size, int reduce, float serial){
    double st, ela;
    st = get_walltime();
    float f_sum = 0;
    float f_max = 0;
    int GRID_DIM = ceil(size/(double)BLOCK_DIM);
    dim3 block_dim(BLOCK_DIM,1,1);
    dim3 grid_dim(GRID_DIM,1,1);
    int share_size;
    if (reduce == 0 || reduce == 1){
        share_size = 2*BLOCK_DIM*sizeof(float);
    }
    else if(reduce == 2){
        share_size = 2*32*sizeof(float);
    }
    int mem_size = GRID_DIM*sizeof(float);
    float *input, *res_sum, *res_max, *cpu_res_sum, *cpu_res_max;
    cudaMalloc((void **) &input, size*sizeof(float));
    cudaMalloc((void **) &res_sum, mem_size);
    cudaMalloc((void **) &res_max, mem_size);
    cpu_res_sum = (float *)malloc(mem_size);
    cpu_res_max = (float *)malloc(mem_size);
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    summax_soft<<<grid_dim, block_dim, share_size>>>(input, size, res_sum, res_max, reduce);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    
    cudaMemcpy(cpu_res_sum, res_sum, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_res_max, res_max, mem_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < GRID_DIM; i++){
        f_sum += cpu_res_sum[i];
        f_max = max_function(cpu_res_max[i], f_max);
    }
    softmax<<<grid_dim, block_dim, share_size>>>(input, f_max, f_sum, size);
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(res_sum);
    cudaFree(res_max);
    free(cpu_res_sum);
    free(cpu_res_max);
    ela = get_walltime() - st;
    printf("reduce:%d,serial:%.3e,parallel sum:%.3e, error:%.3e, max:%.3e\n", reduce, serial, f_sum, f_sum - serial, f_max);
    printf("kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
}

int main() {
    float *cpu_input;
    int size = 16;
    
    float s = 0;
    
    cpu_input = (float *)malloc(size*sizeof(float));
    for(int i = 0; i < size; i++){
        cpu_input[i] = i%100;
        s += cpu_input[i];
    }
    
    
    int reduce;
    
    reduce = 2;
    cpu_summax(cpu_input, size, reduce, s);
    for(int i = 0; i < size; i++){
        printf("softmax:%.3e\n",cpu_input[i]);
    }
    return 0;
}

