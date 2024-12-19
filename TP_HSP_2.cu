#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

dim3 thread_dim (28,28);
dim3 thread_dim2 (14,14);
#define WIDTH 28
#define HEIGHT 28

float *raw_data;
float *C1_data;
float *S1_data;
float *C1_kernel;
float *Mean_kernel;

float *raw_datacuda;
float *C1_datacuda;
float *S1_datacuda;
float *C1_kernelcuda;
float *Mean_kernelcuda;


void MatrixInit_dim2_0_1(float *M, int n, int p) { //matrix n*p with values between 0 and 1
	
	for (int i = 0; i < p; ++i) {
		for (int t =0; t<n;t++) {
			M[i*n+t]=(rand()%101)/100.0f; //between 0 and 1 and i*n+p to index 
		}
	}
}

void MatrixInit_dim3(float *M, int m, int n, int p) { //matrix m n p initialized with 0
	
	for (int i = 0; i < p; ++i) {
		for (int t =0; t<n;t++) {
			for (int z=0; z<m;z++) {
				
				M[i*n+t+z*n*p]=0; //init to 0
			}
		}
	}
}
void MatrixInit_dim3_0_1(float *M, int m, int n, int p) {  //matrix m n p initialized with values between 0 and 1
	
	for (int i = 0; i < p; ++i) {
		for (int t =0; t<n;t++) {
			for (int z=0; z<m;z++) {
				
				M[i*n+t+z*n*p]=(rand()%101)/100.0f; //between 0 and 1 
			}
		}
	}			
}

void ArrayInit(float *M, int size, float value) { 
	for (int i = 0; i < size; ++i) {
		M[i]=value;
	}			
}

void MatrixPrint(float *M, int n, int p) {
	for (int i = 0; i < p; ++i) {
		for (int t =0; t<n;t++) {
			printf("%10.4f  ", M[i*n+t]); //f for float
		}
		printf("\n"); //f for float
	}
}
__global__ void cudaConvolve(float *K, float *I, float *out, int ni, int ki){ //pi number of lines of the images, ni number of columns, K=kernels,I=image, ki=size of the kernel
	int i = blockIdx.x; //kernel number
	int jx = threadIdx.x; //convolve position
	int jy = threadIdx.y; //convolve position

	float conv=0;
	for (int l=0; l<ki;l++) {
		for (int c=0;c<ki;c++) {
			conv+=K[i*ki*ki+l*ki+c]*I[(jx+l)*ni+c+jy];//equation just linearisation de l image.
		}
	}
	out[i * 28 * 28 + jx * 28 + jy] =conv;
}
			
		
__global__ void cudaSampling(float *K, float *I, float *out, int ni, int ki){ //pi number of lines of the images, ni number of columns, K=kernels,I=image, ki=size of the kernel
	int i = blockIdx.x; //kernel number
	int jx = threadIdx.x; //convolve position
	int jy = threadIdx.y; //convolve position

	float mean=0;
	for (int l=0; l<ki;l++) {
		for (int c=0;c<ki;c++) {
			mean+=K[l*ki+c]*I[i*ni*ni+(2*jx+l)*ni+c+2*jy];//equation just linearisation de l image.
		}
	}
	out[i * 14 * 14 + jx * 14 + jy] =mean;
}

__device__ float activation_tanh(float M) {
    return 2.0f / (1.0f + expf(-2.0f * M)) - 1.0f;
}


__global__ void apply_activation_tanh(float *data, int size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = activation_tanh(data[i]);  
    
}

		
	
	
int main() {
	raw_data=(float*)malloc(32*32*sizeof(float));
	C1_data=(float*)malloc(6*28*28*sizeof(float));
	S1_data=(float*)malloc(6*14*14*sizeof(float));
	C1_kernel=(float*)malloc(6*5*5*sizeof(float));
	Mean_kernel=(float*)malloc(2*2*sizeof(float));
	

	
	MatrixInit_dim2_0_1(raw_data,32,32); //matrix 32*32 initialized with values between 0 and 1
	MatrixInit_dim3(C1_data,6,28,28); //matrix 6*28*28 initialized to 0
	MatrixInit_dim3(S1_data,6,14,14); //matrix 6*28*28 initialized to 0
	MatrixInit_dim3_0_1(C1_kernel,6,5,5); //matrix 6*5*5 initialized with values between 0 and 1
	ArrayInit(Mean_kernel,2*2,0.25);//initialisation for mean sampling
	
	//to test
	ArrayInit(raw_data,32*32,0);
	for (int i=0; i<32; i++){
		for (int j=0; j<32; j++){
			if (i == j)
				raw_data[i*32+j]=1;
		}
	}
	
	ArrayInit(C1_data,6*28*28,-1);
	for (int c=0; c<6; c++){
		ArrayInit(C1_kernel+(c*5*5), 5*5, (float)(c+1)/6);
	}
	
	MatrixPrint(raw_data,32,32);
		
	
	cudaMalloc((void**)&raw_datacuda, sizeof(float)*32*32);
    cudaMalloc((void**)&C1_datacuda, sizeof(float)*6*28*28);
    cudaMalloc((void**)&S1_datacuda, sizeof(float)*6*14*14);
    cudaMalloc((void**)&C1_kernelcuda, sizeof(float)*6*5*5);
    cudaMalloc((void**)&Mean_kernelcuda, sizeof(float)*2*2);
    
    cudaMemcpy(raw_datacuda, raw_data, sizeof(float)*32*32, cudaMemcpyHostToDevice); //Cpu to GPU
	cudaMemcpy(C1_datacuda,C1_data , sizeof(float)*6*28*28, cudaMemcpyHostToDevice);
	cudaMemcpy(S1_datacuda, S1_data, sizeof(float)*6*14*14, cudaMemcpyHostToDevice); //Cpu to GPU
	cudaMemcpy(C1_kernelcuda, C1_kernel, sizeof(float)*6*5*5, cudaMemcpyHostToDevice);
	cudaMemcpy(Mean_kernelcuda, Mean_kernel, sizeof(float)*2*2, cudaMemcpyHostToDevice); //Cpu to GPU

	cudaConvolve<<<6,thread_dim>>>(C1_kernelcuda,raw_datacuda,C1_datacuda,32,5);
	cudaSampling<<<6,thread_dim2>>>(Mean_kernelcuda,C1_datacuda,S1_datacuda,28,2);
	//activation_tanh<<<4,294>>>(S1_datacuda); //comment if needed
	
	cudaMemcpy(C1_data, C1_datacuda, sizeof(float)*6*28*28, cudaMemcpyDeviceToHost);
	cudaMemcpy(S1_data, S1_datacuda, sizeof(float)*6*14*14, cudaMemcpyDeviceToHost);
	
	

	for (int i = 0; i < 6; ++i) {
		MatrixPrint(&C1_kernel[i*5*5],5,5);
		printf("convolve \n");
		MatrixPrint(&C1_data[i*28*28],28,28);
	}
	
	for (int i = 0; i < 6; ++i) {
		MatrixPrint(Mean_kernel,2,2);
		printf("Sampling \n");
		MatrixPrint(&S1_data[i*14*14],14,14);
	}
	
	
	free(raw_data);
	free(C1_data);
	free(S1_data);
	free(C1_kernel);
	free(Mean_kernel);
	
	cudaFree(raw_datacuda);
	cudaFree(C1_datacuda);
	cudaFree(S1_datacuda);
	cudaFree(C1_kernelcuda);
	cudaFree(Mean_kernelcuda);
	
	
}
	
	
