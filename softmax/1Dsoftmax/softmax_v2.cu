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
void max_soft(float *input, int size, float *result, int reduce){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (reduce == 0){
        __shared__ float tmp[(int)(BLOCK_DIM)];
        if (i < size){
            tmp[threadIdx.x] = input[i];
        }
        else {
            tmp[threadIdx.x] = -FLT_MAX__;
        }
        __syncthreads();
        for(int strip = 1; strip < blockDim.x; strip = strip*2){
            if (threadIdx.x % (2*strip) == 0){
                tmp[threadIdx.x] = max_function(tmp[threadIdx.x],tmp[threadIdx.x + strip]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            result[blockIdx.x] = tmp[0];
        }
    }
    else if (reduce == 1){
        __shared__ float tmp[(int)(BLOCK_DIM)];
        if (i < size){
            tmp[threadIdx.x] = input[i];
        }
        else {
            tmp[threadIdx.x] = -FLT_MAX__;
        }
        __syncthreads();
        for(int strip = blockDim.x/2; strip > 0; strip = strip/2){
            if (threadIdx.x < strip){
                tmp[threadIdx.x] = max_function(tmp[threadIdx.x],tmp[threadIdx.x + strip]);
            
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            result[blockIdx.x] = tmp[0];
        }
    }
    else if (reduce == 2){
        __shared__ float tmp[32];
        float data = 0.0f;
        float max_data = -FLT_MAX__;
        if (i < size){
            max_data = input[i];
        }
        for(int offset = 16; offset >0; offset/= 2){
            data = __shfl_down_sync(0xffffffff, max_data, offset);
            max_data = max_function(max_data, data);
        }

        if (threadIdx.x % 32 == 0){
            tmp[threadIdx.x/32] = max_data;
        }
        __syncthreads();
        if(threadIdx.x >= 32)
            return;
        max_data = tmp[threadIdx.x];
        for(int offset = 16; offset >0; offset/= 2){
            data = __shfl_down_sync(0xffffffff, max_data, offset);
            max_data = max_function(max_data, data);
        }
        if (threadIdx.x == 0){
            result[blockIdx.x] = max_data;
        }
    }
}
__global__ void gridmax(float *result, int num_blocks){//result[0] = max
    for(int stride = num_blocks/2; stride > 0; stride = stride/2){
        if (threadIdx.x < stride) {//threadIdx.x + stride < num_blocks = len(result)
            result[threadIdx.x] = max_function(result[threadIdx.x + stride], result[threadIdx.x]);
        }
        __syncthreads();
    }
}

__global__
void sum_soft(float *input, int size, float *result, int reduce){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < size){
        input[i] = __expf(input[i] - result[0]);
    }
    if (reduce == 0){
        __shared__ float tmp[(int)(BLOCK_DIM)];
        if (i < size){
            tmp[threadIdx.x] = input[i];
        }
        else {
            tmp[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for(int strip = 1; strip < blockDim.x; strip = strip*2){
            if (threadIdx.x % (2*strip) == 0){
                tmp[threadIdx.x] += tmp[threadIdx.x + strip];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            result[blockIdx.x] = tmp[0];
        }
    }
    else if (reduce == 1){
        __shared__ float tmp[(int)(BLOCK_DIM)];
        if (i < size){
            tmp[threadIdx.x] = input[i];
        }
        else {
            tmp[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for(int strip = blockDim.x/2; strip > 0; strip = strip/2){
            if (threadIdx.x < strip){
                tmp[threadIdx.x] += tmp[threadIdx.x + strip];
            
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            result[blockIdx.x] = tmp[0];
        }
    }
    else if (reduce == 2){
        __shared__ double tmp[32];
        double data = 0.0f;
        if (i < size){
            data = input[i];
        }
        data += __shfl_down_sync(0xffffffff, data, 16);// 0 + 16, 1 + 17,..., 15 + 31
        data += __shfl_down_sync(0xffffffff, data, 8);// 0 + 8, 1 + 9,..., 7 + 15
        data += __shfl_down_sync(0xffffffff, data, 4);
        data += __shfl_down_sync(0xffffffff, data, 2);
        data += __shfl_down_sync(0xffffffff, data, 1);
        if (threadIdx.x % 32 == 0){
            tmp[threadIdx.x/32] = data;
        }
        __syncthreads();
        if(threadIdx.x >= 32)
            return;
        data = tmp[threadIdx.x];
        data += __shfl_down_sync(0xffffffff, data, 16);// 0 + 16, 1 + 17,..., 15 + 31
        data += __shfl_down_sync(0xffffffff, data, 8);// 0 + 8, 1 + 9,..., 7 + 15
        data += __shfl_down_sync(0xffffffff, data, 4);
        data += __shfl_down_sync(0xffffffff, data, 2);
        data += __shfl_down_sync(0xffffffff, data, 1);
        if (threadIdx.x == 0){
            result[blockIdx.x] = data;
        }
    }
}
__global__ void gridsum(float *result, int num_blocks){//result[0] = sum
    for(int stride = num_blocks/2; stride > 0; stride = stride/2){
        if (threadIdx.x < stride) {
            result[threadIdx.x] += result[threadIdx.x + stride];
        }
        __syncthreads();
    }
}
__global__ void softmax(float *input, float *result, int size){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < size){
        input[i] /= result[0];
    }
}
void cpu_softmax(float *cpu_input, int size, int reduce){
    double st, ela;
    st = get_walltime();
    
    int num_blocks = ceil(size/(double)BLOCK_DIM);
    dim3 block_dim(BLOCK_DIM,1,1);
    dim3 grid_dim(num_blocks,1,1);
    int share_size;
    if (reduce == 0 || reduce == 1){
        share_size = BLOCK_DIM*sizeof(float);
    }
    else if(reduce == 2){
        share_size = 32*sizeof(float);
    }
    int mem_size = num_blocks*sizeof(float);
    float *input, *result, *cpu_result;
    cudaMalloc((void **) &input, size*sizeof(float));
    cudaMalloc((void **) &result, mem_size);
    cpu_result = (float *)malloc(mem_size);
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    max_soft<<<grid_dim, block_dim, share_size>>>(input, size, result, reduce);
    cudaDeviceSynchronize();
    gridmax<<<grid_dim, block_dim>>>(result, num_blocks);
    cudaDeviceSynchronize();
    sum_soft<<<grid_dim, block_dim, share_size>>>(input, size, result, reduce);
    cudaDeviceSynchronize();
    gridsum<<<grid_dim, block_dim>>>(result, num_blocks);
    cudaDeviceSynchronize();
    softmax<<<grid_dim, block_dim>>>(input, result, size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(result);
    free(cpu_result);
    ela = get_walltime() - st;
    
    printf("reduce:%d,kernel time:%.4f, use time:%.4f\n", reduce, ker_time/1000., ela);
    
}

int main() {
    float *cpu_input;
    int size = 1024*200000;
    
    cpu_input = (float *)malloc(size*sizeof(float));
    for(int i = 0; i < size; i++){
        cpu_input[i] = i%100;
        
    }
    
    int reduce;
    for(reduce = 0; reduce < 3; reduce++){
        cpu_softmax(cpu_input, size, reduce);
    }
    
    
    /***
     * float s = 0;
    for(int i = 0; i < size; i++){
        s += cpu_input[i];
        printf("softmax:%.3e\n",cpu_input[i]);
    }
    printf("s:%.3e\n",s);
    ***/
    free(cpu_input);
    
    
    return 0;
}




