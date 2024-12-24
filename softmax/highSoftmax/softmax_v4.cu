#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>
#define BLOCK_DIM_y 32
#define BLOCK_DIM_x 32
#define max_function(a,b) ((a)>(b)?(a):(b)) 

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}

__global__ void _softmax_kernel(float *__restrict input,
                         int size, int dimsize,
                         int stride) {  // if set axis = 1
    int i = threadIdx.x + blockIdx.x * blockDim.x; // i < inputShape[axis]
    int j = threadIdx.y + blockIdx.y * blockDim.y; // j < size/inputShape[axis]
    int size_x = dimsize;
    int size_y = size/size_x;
    int tid = j % stride + (j - j % stride) * size_x;
    
    __shared__ float res_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float res_max[BLOCK_DIM_x][BLOCK_DIM_y];   
    
    res_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    res_sum[threadIdx.x][threadIdx.y] = 0.0f;
    __syncthreads();
    for (int ph = 0; threadIdx.x + ph*blockDim.x < dimsize; ph++) {
        if(res_max[threadIdx.x][threadIdx.y] > input[tid + (threadIdx.x + ph*blockDim.x) * stride]){
            res_sum[threadIdx.x][threadIdx.y] = res_sum[threadIdx.x][threadIdx.y] + __expf(input[tid + (threadIdx.x + ph*blockDim.x) * stride] - res_max[threadIdx.x][threadIdx.y]);  
        }
        else{
            res_sum[threadIdx.x][threadIdx.y] = 1 + res_sum[threadIdx.x][threadIdx.y]*__expf(res_max[threadIdx.x][threadIdx.y] - input[tid + (threadIdx.x + ph*blockDim.x) * stride]);
            res_max[threadIdx.x][threadIdx.y] = input[tid + (threadIdx.x + ph*blockDim.x) * stride];
        }
    }
    __syncthreads();
    for(int strip = blockDim.x/2; strip > 0; strip = strip/2){
        if (threadIdx.x < strip){
            if(res_max[threadIdx.x][threadIdx.y] > res_max[threadIdx.x + strip][threadIdx.y]){
                res_sum[threadIdx.x][threadIdx.y] = res_sum[threadIdx.x][threadIdx.y] + res_sum[threadIdx.x + strip][threadIdx.y]*__expf(res_max[threadIdx.x + strip][threadIdx.y] - res_max[threadIdx.x][threadIdx.y]);
            }
        else{
            res_sum[threadIdx.x][threadIdx.y] = res_sum[threadIdx.x + strip][threadIdx.y] + res_sum[threadIdx.x][threadIdx.y]*__expf(res_max[threadIdx.x][threadIdx.y] - res_max[threadIdx.x + strip][threadIdx.y]);
            res_max[threadIdx.x][threadIdx.y] = res_max[threadIdx.x + strip][threadIdx.y];
            }
        }
    }
    __syncthreads();
        //-----------------
    if(i < dimsize && j < size_y) {
        //output[tid + i * stride] = __expf(share_input[threadIdx.x][threadIdx.y] - res_max[threadIdx.y]) * __fdividef(1.0F, res_sum[threadIdx.y]);
        input[tid + i * stride] = __expf(input[tid + i * stride] - res_max[0][threadIdx.y]) *  __fdividef(1.0F, res_sum[0][threadIdx.y]);
    } 
}
void softmax(float *cpu_input, int size, int *cpu_inputShape, int axis, int nDims, int stride){
    double st, ela;
    st = get_walltime();
    int size_x = cpu_inputShape[axis];
    int size_y = size/cpu_inputShape[axis];
    
    int num_block_x = ceil(size_x/(double)BLOCK_DIM_x);
    int num_block_y = ceil(size_y/(double)BLOCK_DIM_y);
    dim3 block_dim(BLOCK_DIM_x,BLOCK_DIM_y,1);
    dim3 grid_dim(num_block_x,num_block_y,1);
    
    
    float *input, *res_sum, *res_max;
    cudaMalloc((void **) &input, size*sizeof(float));
    cudaMalloc((void **) &res_sum, size_y*sizeof(float));
    cudaMalloc((void **) &res_max, size_y*sizeof(float));
    cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
    int *inputShape;
    cudaMalloc((void **) &inputShape, nDims*sizeof(float));
    cudaMemcpy(inputShape, cpu_inputShape, nDims*sizeof(float), cudaMemcpyHostToDevice);
    int share_mem = (2*BLOCK_DIM_y + 2)*sizeof(float); 
    
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    _softmax_kernel<<<grid_dim, block_dim, share_mem>>>(input, size, size_x, stride);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(inputShape);
    cudaFree(res_sum);
    cudaFree(res_max);
    ela = get_walltime() - st;
    
    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
    
}
int main() {
    
    int nDims = 4;
    int cpu_inputShape[nDims] = {4,512,256,1024};
    int axis = 1;
    
    int size = 1;
    int stride = 1;
    for(int i = nDims - 1; i >= 0; --i){
        if(i == axis){
            stride = size;
        }
        size *= cpu_inputShape[i];
        
    }
    float *cpu_input;
    cpu_input = (float *)malloc(size*sizeof(float));
    for(int i = 0; i < size; i++){
        cpu_input[i] = i%10;
    }
    
    softmax(cpu_input, size, cpu_inputShape, axis, nDims, stride);
    
    float s = 0;
    
    for(int i = 0; i < size; i++){
        s += cpu_input[i];
        //printf("softmax:%.4e\n",cpu_input[i]);
    }
    printf("s:%.3e\n",s);
    free(cpu_input);
    
    
    return 0;
}






