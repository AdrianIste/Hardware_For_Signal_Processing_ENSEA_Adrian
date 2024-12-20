#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
float *M1;
float *M2;
float *Mout;
float *Mout2;
float *MoutMult;
float *MoutMult2;

float *M1cuda;
float *M2cuda;
float *Moutcuda;
float *MoutMultcuda;

int n; //number of lines
int p; //number of columns

int grid_size;
int block_size;

void MatrixInit(float *M, int n, int p) { //we put ** because it works
	
	for (int i = 0; i < p; ++i) {
		for (int t =0; t<n;t++) {
			M[i*n+t]=(rand()%201-100)/100.0f; //between -1 and 1 and i*n+p to index 
		}
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

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) { //matrix addition
	for (int i = 0; i < p; ++i) {
		for (int t =0; t<n;t++) {
			Mout[i*n+t]=M1[i*n+t]+M2[i*n+t];
		}
	}
}


void MatrixMult(float *M1, float *M2, float *Mout, int n) { //matrix multiplication
	float temp;
	for (int i = 0; i < n; ++i) {
		for (int j=0; j<n;j++) {
			
			temp=0; //to compute the addition for each line and columns
			for (int t =0; t<n;t++) {
				temp+=M1[i*n+t]*M2[t*n+j];
			}
		Mout[i*n+j]=temp;
	}
	}
}
			
			
		
		
		
	
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){ //addition on cuda
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid<n*p) {
		Mout[tid]=M1[tid]+M2[tid];
		}
	
}
	
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) { //mult on cuda
	int i = blockIdx.x; //line
	int j = threadIdx.x; //column

	float temp=0; //to compute the addition for each line and columns
	for (int t =0; t<n;t++) {
		temp+=M1[i*n+t]*M2[t*n+j];
			}
		Mout[i*n+j]=temp;
}
		

		
			
			
			
int main(int argc, char *argv[]) {
	n=atoi(argv[1]);
	p=atoi(argv[2]);
	grid_size=atoi(argv[3]);
	block_size=atoi(argv[4]);
	M1=(float*)malloc(n*p*sizeof(float));
	M2=(float*)malloc(n*p*sizeof(float));
	Mout=(float*)malloc(n*p*sizeof(float));
	Mout2=(float*)malloc(n*p*sizeof(float));

	MoutMult=(float*)malloc(n*n*sizeof(float));
	MoutMult2=(float*)malloc(n*n*sizeof(float));
	
	MatrixInit(M1,n,p);
	MatrixInit(M2,n,p);
	clock_t start = clock();
	MatrixAdd(M1,M2,Mout,n,p);
	clock_t end = clock();
	
	clock_t start2 = clock();
	MatrixMult(M1,M2,MoutMult,n);
	clock_t end2 = clock();
	float elapsed_time = (float)(end - start) / CLOCKS_PER_SEC; 
	float elapsed_time2 = (float)(end2 - start2) / CLOCKS_PER_SEC; 

	
	printf("Matrice M1 :\n");
	MatrixPrint(M1,n,p);
	printf("Matrice M2 :\n");
	MatrixPrint(M2,n,p);
	printf("Matrice Mout :\n");
	MatrixPrint(Mout,n,p);
	printf("MatrixAdd Time: %f seconds\n", elapsed_time);
	printf("Matrice MoutMult :\n");
	MatrixPrint(MoutMult,n,n);
	printf("MatrixMult Time: %f seconds\n", elapsed_time2);
	cudaMalloc((void**)&M1cuda, sizeof(float)*grid_size*block_size);
    cudaMalloc((void**)&M2cuda, sizeof(float)*grid_size*block_size);
    cudaMalloc((void**)&Moutcuda, sizeof(float)*grid_size*block_size);
    cudaMalloc((void**)&MoutMultcuda, sizeof(float)*grid_size*block_size);
	

	
	cudaMemcpy(M1cuda, M1, sizeof(float)*grid_size*block_size, cudaMemcpyHostToDevice); //Cpu to GPU
	cudaMemcpy(M2cuda, M2, sizeof(float)*grid_size*block_size, cudaMemcpyHostToDevice);
	
	clock_t start3 = clock();

    cudaMatrixAdd<<<grid_size,block_size>>>(M1cuda,M2cuda,Moutcuda,grid_size,block_size);
    clock_t end3 = clock();

    clock_t start4 = clock();

    cudaMatrixMult<<<grid_size,block_size>>>(M1cuda,M2cuda,MoutMultcuda,grid_size);

   	clock_t end4 = clock();

    cudaMemcpy(Mout2, Moutcuda, sizeof(float)*grid_size*block_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(MoutMult2, MoutMultcuda, sizeof(float)*grid_size*block_size, cudaMemcpyDeviceToHost);
	float elapsed_time3 = (float)(end3 - start3) / CLOCKS_PER_SEC; 
	float elapsed_time4 = (float)(end4- start4) / CLOCKS_PER_SEC; 

    
	printf("Matrice MoutAddCuda :\n");
	MatrixPrint(Mout2,grid_size,block_size);
	printf("MatrixAdd cuda Time: %f seconds\n", elapsed_time3);

	printf("Matrice MoutMultCuda :\n");
	MatrixPrint(MoutMult2,grid_size,block_size);
	printf("MatrixMult cuda Time: %f seconds\n", elapsed_time4);

	
    
    
    
	free(M1);
	free(M2);
	free(Mout);
	free(Mout2);
	free(MoutMult);
	free(MoutMult2);

	cudaFree(M1cuda);
    cudaFree(M2cuda);
    cudaFree(Moutcuda);
    cudaFree(MoutMultcuda);

	
	
	
	
	return 0;

}
	
