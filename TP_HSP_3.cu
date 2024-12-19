#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

dim3 thread_dim (28,28);
dim3 thread_dim2 (14,14);
dim3 thread_dim3 (10,10);
dim3 thread_dim4 (5,5);

#define WIDTH 28
#define HEIGHT 28

float *raw_data;
float *C1_data;
float *C2_data;

float *S1_data;
float *S2_data;
float *D1_data;
float *D2_data;
float *D3_data;
float *D1_weights;
float *D2_weights;
float *D3_weights;

float *C1_kernel;
float *C2_kernel;

float *Mean_kernel;
float *Bias_0;
float *Bias_1;
float *Bias_2;
float *Bias_3;
float *Bias_4;

float *raw_datacuda;
float *C1_datacuda;
float *C2_datacuda;

float *S1_datacuda;
float *S2_datacuda;
float *C1_kernelcuda;
float *C2_kernelcuda;
float *D1_datacuda;
float *D2_datacuda;
float *D3_datacuda;
float *D1_weightscuda;
float *D2_weightscuda;
float *D3_weightscuda;
float *Mean_kernelcuda;
float *Bias_0cuda;
float *Bias_1cuda;
float *Bias_2cuda;
float *Bias_3cuda;
float *Bias_4cuda;

unsigned int readBigEndianInt(FILE *fptr) {
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fptr);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

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
void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
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

/*void load_weights(const char* filename, float *weights, int m, int n, int p) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
        printf("Unable to open weight file\n");
        exit(1);
    }

    // Lecture des poids dans la matrice 3D (m x n x p)
    for (int i = 0; i < p; i++) {  // Pour chaque noyau (p = 6 dans votre cas)
        for (int j = 0; j < n; j++) {  // Pour chaque ligne (n = 5)
            for (int k = 0; k < m; k++) {  // Pour chaque colonne (m = 5)
                if (fscanf(fptr, "%f", &weights[(i * n + j) * m + k]) != 1) {
                    printf("Error reading weight values\n");
                    fclose(fptr);
                    exit(1);
                }
            }
        }
    }
}*/
void load_weights(const char* filename, float *weights, int m, int n, int num_kernels, int depth) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
        printf("Unable to open weight file: %s\n", filename);
        exit(1);
    }

    // Lecture des poids dans une matrice 4D (num_kernels x n x m x depth)
	for (int j = 0; j < n; j++) {       // Parcourir les lignes
        for (int i = 0; i < m; i++) {   // Parcourir les colonnes
			for (int d = 0; d < depth; d++) { // Parcourir les profondeurs

                for (int k = 0; k < num_kernels; k++) { // Parcourir les noyaux		

                    if (fscanf(fptr, "%f", &weights[k*m*n*depth+d*m*n+j*m+i]) != 1) {
                        printf("Error reading weight values\n");
                        fclose(fptr);
                        exit(1);
                    }
                }
            }
        }
    }

    fclose(fptr);
}
void load_weights2(const char* filename, float *weights, int m, int n, int num_kernels, int depth) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
        printf("Unable to open weight file: %s\n", filename);
        exit(1);
    }

    // Lecture des poids dans une matrice 4D (num_kernels x n x m x depth)
	for (int j = 0; j < n; j++) {       // Parcourir les lignes
        for (int i = 0; i < m; i++) {   // Parcourir les colonnes
			for (int d = 0; d < depth; d++) { // Parcourir les profondeurs

                for (int k = 0; k < num_kernels; k++) { // Parcourir les noyaux		

                    if (fscanf(fptr, "%f", &weights[k*m*n*depth+d*m*n+j*m+i]) != 1) {
                        printf("Error reading weight values\n");
                        fclose(fptr);
                        exit(1);
                    }
                    int index = k * m * n * depth + d * m * n + j * m + i;
                    printf("k=%d, d=%d, j=%d, i=%d => Index 1D = %d\n", k, d, j, i, index);

                }
            }
        }
    }

    fclose(fptr);
}

/*void load_weights(const char* filename, float *weights, int m, int n, int p, int q) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
        printf("Unable to open weight file\n");
        exit(1);
    }

    // Lecture des poids dans la matrice 4D (m x n x p x q)
    for (int i = 0; i < p; i++) {  // Pour chaque noyau (p noyaux)
        for (int j = 0; j < n; j++) {  // Pour chaque ligne (n)
            for (int k = 0; k < m; k++) {  // Pour chaque colonne (m)
                for (int l = 0; l < q; l++) {  // Pour chaque profondeur (q couches)
                    // Calcul de l'indice dans la matrice 4D
                    int index = ((i * n + j) * m + k) * q + l;
                    if (fscanf(fptr, "%f", &weights[index]) != 1) {
                        printf("Error reading weight values\n");
                        fclose(fptr);
                        exit(1);
                    }
                }
            }
        }
    }

    fclose(fptr);
}*/

    
    
__global__ void cudaConvolve(float *K, float *I, float *out, int ni, int ki, int hi){ //pi number of lines of the images, ni number of columns, K=kernels,I=image, ki=size of the kernel, h depth of input
	int i = blockIdx.x; //kernel number
	int jx = threadIdx.x; //convolve position
	int jy = threadIdx.y; //convolve position

	float conv=0;
	for (int l=0; l<ki;l++) {
		for (int c=0;c<ki;c++) {
			for (int h=0; h<hi;h++) {
				conv+=K[i*ki*ki+l*ki+c]*I[(h-1)*hi*hi+(jx+l)*ni+c+jy];//equation just linearisation de l image.
			}
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

__global__ void add_bias(float *B, float *I, float *out){ //pi number of lines of the images, ni number of columns, B=bias,I=image, ki=size of the kernel
	int i = blockIdx.x; //bias number
	int j=threadIdx.x + blockIdx.x * blockDim.x;

	out[j]=I[j]+B[i];
		
	
}

__global__ void cudaDense(float *in, float *out, float *B,float *w, int n_in) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	float add=0;
	for (int j=0;j<n_in;j++) {
		 
		add+=in[j]*w[i*n_in+j];
	}
	add+=B[i];
	out[i]=add;
}

		
	
	
int main() {
	
    // Allocation de la mémoire pour les données
    raw_data = (float*)malloc(32 * 32 * sizeof(float));
    C1_data = (float*)malloc(6 * 28 * 28 * sizeof(float));
    S1_data = (float*)malloc(6 * 14 * 14 * sizeof(float));
    S2_data = (float*)malloc(16 * 5 * 5 * sizeof(float));
    C2_data = (float*)malloc(16 * 10 * 10 * sizeof(float));
    D1_data = (float*)malloc(120 * sizeof(float));
    D2_data = (float*)malloc(84 * sizeof(float));
    D3_data = (float*)malloc(10 * sizeof(float));
    C1_kernel = (float*)malloc(6 * 5 * 5 * sizeof(float));  // Pour 6 noyaux 5x5
    C2_kernel = (float*)malloc(16 * 5 * 5 * sizeof(float));
    Mean_kernel = (float*)malloc(2 * 2 * sizeof(float));
	Bias_0=(float*)malloc(6*1*sizeof(float));
	Bias_1=(float*)malloc(16*1*sizeof(float));
	Bias_2=(float*)malloc(120*1*sizeof(float));
	Bias_3=(float*)malloc(84*1*sizeof(float));
	Bias_4=(float*)malloc(10*1*sizeof(float));
	D1_weights = (float*)malloc(400*120 * sizeof(float));
	D2_weights = (float*)malloc(120*84 * sizeof(float));
	D3_weights = (float*)malloc(84*10 * sizeof(float));
    // Initialisation des matrices
    int i, j;
    int ***img;
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;
    int imageIndex = 2;  // Indice de l'image à lire (commence à 1)

    // Ouvrir le fichier IDX3
    if ((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL) {
        printf("Impossible d'ouvrir le fichier\n");
        exit(1);
    }

    // Lire l'en-tête
    magic = readBigEndianInt(fptr);
    nbImg = readBigEndianInt(fptr);
    nbRows = readBigEndianInt(fptr);
    nbCols = readBigEndianInt(fptr);

    printf("Magic Number : %u\n", magic);
    printf("Number of Images : %u\n", nbImg);
    printf("Number of Rows : %u\n", nbRows);
    printf("Number of Columns : %u\n", nbCols);

    if (imageIndex > nbImg) {
        printf("L'indice de l'image dépasse le nombre total d'images disponibles\n");
        exit(1);
    }

    // Allouer de la mémoire pour l'image (28x28 pixels avec 3 couleurs)
    img = (int ***)malloc(HEIGHT * sizeof(int **));
    for (i = 0; i < HEIGHT; i++) {
        img[i] = (int **)malloc(WIDTH * sizeof(int *));
        for (j = 0; j < WIDTH; j++) {
            img[i][j] = (int *)malloc(sizeof(int) * 3);
        }
    }

    // Allouer de la mémoire pour le tableau 1D (niveaux de gris)

    // Positionner le pointeur sur l'image désirée
    fseek(fptr, (imageIndex - 1) * nbRows * nbCols, SEEK_CUR);

    // Lire l'image et convertir en couleurs
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            fread(&val, sizeof(unsigned char), 1, fptr);
            img[i][j][0] = (int)val; // Valeur rouge
            img[i][j][1] = (int)val; // Valeur verte
            img[i][j][2] = (int)val; // Valeur bleue

            // Conversion en niveaux de gris
            raw_data[i * WIDTH + j] = (float)val / 255.0f;  // Normalisé entre 0 et 1
        }
    }

    // Afficher l'image en couleur
    imgColorPrint(HEIGHT, WIDTH, img);

    // Afficher le tableau 1D (niveaux de gris normalisés)
    printf("Image sous forme de tableau 1D (niveaux de gris normalisés) :\n");
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            printf("%10.4f ", raw_data[i * WIDTH + j]);
        }
        printf("\n");
    }

    // Libérer la mémoire
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            free(img[i][j]);
        }
        free(img[i]);
    }
    free(img);

    //exit(EXIT_SUCCESS);

    //MatrixInit_dim2_0_1(raw_data, 32, 32);
    MatrixInit_dim2_0_1(D1_data, 1, 120);
    MatrixInit_dim2_0_1(D2_data, 1, 84);
    MatrixInit_dim2_0_1(D3_data, 1, 10);
    
    MatrixInit_dim3(C1_data, 6, 28, 28);
    MatrixInit_dim3(S1_data, 6, 14, 14);
    ArrayInit(Mean_kernel, 2 * 2, 0.25);

    // Charger les poids des noyaux de convolution
    load_weights("layer_0_weights.txt", C1_kernel, 5, 5, 6,1);  // Charger 6 noyaux 5x5
    load_weights("layer_0_bias.txt", Bias_0,6,1,1,1);
    load_weights("layer_1_weights.txt", C2_kernel, 5, 5, 16,6);  // Charger 6 noyaux 5x5
    load_weights("layer_1_bias.txt", Bias_1,16,1,1,1);
    load_weights("layer_2_weights.txt", D1_weights,1,400,120,1);
    load_weights("layer_2_bias.txt", Bias_2,120,1,1,1);
    load_weights("layer_3_weights.txt", D2_weights,1,120,84,1);
    load_weights("layer_3_bias.txt", Bias_3,84,1,1,1);
    load_weights("layer_4_weights.txt", D3_weights,1,84,10,1);
    load_weights("layer_4_bias.txt", Bias_4,10,1,1,1);
    // Pour vérifier si les poids sont correctement chargés
    for (int i = 0; i < 6; ++i) {
        MatrixPrint(&C1_kernel[i * 5 * 5], 5, 5);
        printf("Kernel %d loaded\n", i + 1);
    }
	for (int i = 0; i < 16; ++i) {  // Parcourir tous les noyaux de la couche C2
		printf("Kernel %d (C2) loaded:\n", i + 1);

		// Afficher chaque couche de 5x5 pour ce noyau
		for (int d = 0; d < 6; ++d) {  // Parcourir chaque profondeur (canal d'entrée)
			printf("  Depth %d:\n", d + 1);  // Indiquer la couche actuelle
			MatrixPrint(&C2_kernel[i * 5 * 5 * 6 + d * 5 * 5], 5, 5);  // Afficher la matrice 5x5 pour ce canal
		}
	}
    // Allocation et copie sur le GPU
    cudaMalloc((void**)&raw_datacuda, sizeof(float) * 32 * 32);
    cudaMalloc((void**)&C1_datacuda, sizeof(float) * 6 * 28 * 28);
    cudaMalloc((void**)&S1_datacuda, sizeof(float) * 6 * 14 * 14);
    cudaMalloc((void**)&C1_kernelcuda, sizeof(float) * 6 * 5 * 5);
    cudaMalloc((void**)&C2_kernelcuda, sizeof(float) * 16 * 5 * 5*6);
    cudaMalloc((void**)&Mean_kernelcuda, sizeof(float) * 2 * 2);
    cudaMalloc((void**)&Bias_0cuda,sizeof(float)*6);
    cudaMalloc((void**)&Bias_1cuda,sizeof(float)*16);
    cudaMalloc((void**)&Bias_2cuda,sizeof(float)*120);
    cudaMalloc((void**)&Bias_3cuda,sizeof(float)*84);
    cudaMalloc((void**)&Bias_4cuda,sizeof(float)*10);
	cudaMalloc((void**)&S2_datacuda, 16 * 5 * 5 * sizeof(float));  // Allocation pour S2_data
	cudaMalloc((void**)&C2_datacuda, 16 * 10 * 10 * sizeof(float)); // Allocation pour C2_data
	cudaMalloc((void**)&D1_datacuda, 120 * sizeof(float));          // Allocation pour D1_data
	cudaMalloc((void**)&D2_datacuda, 84 * sizeof(float));           // Allocation pour D2_data
	cudaMalloc((void**)&D3_datacuda, 10 * sizeof(float)); 
	cudaMalloc((void**)&D1_weightscuda, sizeof(float) * 120*400);
	cudaMalloc((void**)&D2_weightscuda, sizeof(float) * 84*120);
	cudaMalloc((void**)&D3_weightscuda, sizeof(float) * 10*84);	
	
	cudaMemcpy(D1_weightscuda, D1_weights, sizeof(float) * 120*400, cudaMemcpyHostToDevice);
	cudaMemcpy(D2_weightscuda, D2_weights, sizeof(float) * 84*120, cudaMemcpyHostToDevice);
	cudaMemcpy(D3_weightscuda, D3_weights, sizeof(float) * 10*84, cudaMemcpyHostToDevice);	
    cudaMemcpy(raw_datacuda, raw_data, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_datacuda, C1_data, sizeof(float) * 6 * 28 * 28, cudaMemcpyHostToDevice);
    cudaMemcpy(S1_datacuda, S1_data, sizeof(float) * 6 * 14 * 14, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernelcuda, C1_kernel, sizeof(float) * 6 * 5 * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(Mean_kernelcuda, Mean_kernel, sizeof(float) * 2 * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(S2_datacuda, S2_data, sizeof(float) * 16 * 5 * 5, cudaMemcpyHostToDevice);
	cudaMemcpy(C2_datacuda, C2_data, sizeof(float) * 16 * 10 * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(D1_datacuda, D1_data, sizeof(float) * 120, cudaMemcpyHostToDevice);
	cudaMemcpy(D2_datacuda, D2_data, sizeof(float) * 84, cudaMemcpyHostToDevice);
	cudaMemcpy(D3_datacuda, D3_data, sizeof(float) * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(C2_kernelcuda, C2_kernel, sizeof(float) * 16 * 5 * 5*6, cudaMemcpyHostToDevice);
	cudaMemcpy(Bias_0cuda, Bias_0, sizeof(float)*6, cudaMemcpyHostToDevice);
	cudaMemcpy(Bias_1cuda, Bias_1, sizeof(float)*16, cudaMemcpyHostToDevice);
	cudaMemcpy(Bias_2cuda, Bias_2, sizeof(float)*120, cudaMemcpyHostToDevice);
	cudaMemcpy(Bias_3cuda, Bias_3, sizeof(float)*84, cudaMemcpyHostToDevice);
	cudaMemcpy(Bias_4cuda, Bias_4, sizeof(float)*10, cudaMemcpyHostToDevice);


    // Convolution et sampling sur GPU
    cudaConvolve<<<6, thread_dim>>>(C1_kernelcuda, raw_datacuda, C1_datacuda, 32, 5,1);
    add_bias<<<6, 784>>>(Bias_0cuda, C1_datacuda, C1_datacuda);
    apply_activation_tanh<<<6, 784>>>(C1_datacuda, 6 * 784);

    cudaSampling<<<6, thread_dim2>>>(Mean_kernelcuda, C1_datacuda, S1_datacuda, 28, 2);
    
    cudaConvolve<<<16, thread_dim3>>>(C2_kernelcuda, S1_datacuda, C2_datacuda, 14, 5,6);
    add_bias<<<16, 100>>>(Bias_1cuda, C2_datacuda, C2_datacuda);
    apply_activation_tanh<<<16, 100>>>(C2_datacuda, 16*100);

    cudaSampling<<<16, thread_dim4>>>(Mean_kernelcuda, C2_datacuda, S2_datacuda, 10, 2);

    
    
    cudaDense<<<1,120>>>(S2_datacuda, D1_datacuda,Bias_2cuda,D1_weightscuda, 400);
    apply_activation_tanh<<<1, 120>>>(D1_datacuda,  120);
    
    cudaDense<<<1,84>>>(D1_datacuda, D2_datacuda,Bias_3cuda,D2_weightscuda, 120);
    apply_activation_tanh<<<1, 84>>>(D2_datacuda, 84);
    
    cudaDense<<<1,10>>>(D2_datacuda, D3_datacuda,Bias_4cuda,D3_weightscuda, 84);
    
    apply_activation_tanh<<<1, 10>>>(D3_datacuda, 10);
  
    
    
    
    
    

    // Copier les résultats du GPU vers la mémoire CPU
    cudaMemcpy(C1_data, C1_datacuda, sizeof(float) * 6 * 28 * 28, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, S1_datacuda, sizeof(float) * 6 * 14 * 14, cudaMemcpyDeviceToHost);
	cudaMemcpy(C2_data, C2_datacuda, sizeof(float) * 16 * 10 * 10, cudaMemcpyDeviceToHost);

// Copier les résultats après l'échantillonnage S2
	cudaMemcpy(S2_data, S2_datacuda, sizeof(float) * 16 * 5 * 5, cudaMemcpyDeviceToHost);

	// Copier les résultats après la couche dense D1
	cudaMemcpy(D1_data, D1_datacuda, sizeof(float) * 120, cudaMemcpyDeviceToHost);

	// Copier les résultats après la couche dense D2
	cudaMemcpy(D2_data, D2_datacuda, sizeof(float) * 84, cudaMemcpyDeviceToHost);

	// Copier les résultats après la couche dense D3 (sortie finale)
	cudaMemcpy(D3_data, D3_datacuda, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    // Affichage des résultats
  /*  for (int i = 0; i < 6; ++i) {
        MatrixPrint(&C1_data[i * 28 * 28], 28, 28);
        printf("Convolution result for Kernel(with bias added) %d\n", i + 1);
    }

    for (int i = 0; i < 6; ++i) {
        MatrixPrint(&S1_data[i * 14 * 14], 14, 14);
        printf("Sampling result for Kernel %d\n", i + 1);
    } */
    printf("Convolution result for C1:\n");
	for (int i = 0; i < 6; ++i) {
		MatrixPrint(&C1_data[i * 28 * 28], 28, 28); // Afficher la sortie de C1
		printf("Convolution result for Kernel(with bias added) %d\n", i + 1);
	}

	// Affichage du résultat de l'échantillonnage S1
	printf("Sampling result for S1:\n");
	for (int i = 0; i < 6; ++i) {
		MatrixPrint(&S1_data[i * 14 * 14], 14, 14); // Afficher la sortie de S1
        printf("Sampling result for Kernel %d\n", i + 1);
		
	}

	// Affichage du résultat de la convolution C2
	printf("Convolution result for C2:\n");
	for (int i = 0; i < 16; ++i) {
		MatrixPrint(&C2_data[i * 10 * 10], 10, 10); // Afficher la sortie de C2
		printf("Convolution result for Kernel(with bias added) %d\n", i + 1);

	}

	// Affichage du résultat de l'échantillonnage S2
	printf("Sampling result for S2:\n");
	for (int i = 0; i < 16; ++i) {
		MatrixPrint(&S2_data[i * 5 * 5], 5, 5); // Afficher la sortie de S2
        printf("Sampling result for Kernel %d\n", i + 1);

	}

	// Affichage du résultat de D1 après dense layer et activation
	printf("Dense layer result for D1:\n");
	MatrixPrint(D1_data, 120, 1); // Afficher la sortie de D1

	// Affichage du résultat de D2 après dense layer et activation
	printf("Dense layer result for D2:\n");
	MatrixPrint(D2_data, 84, 1); // Afficher la sortie de D2

	// Affichage du résultat de D3 après dense layer et activation
	printf("Dense layer result for D3:\n");
	MatrixPrint(D3_data, 10, 1); // Afficher la sortie de D3

    // Libération de la mémoire
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(Mean_kernel);
    free(Bias_0);
    free(Bias_1);
    free(Bias_2);
    free(Bias_3);
    free(Bias_4);
	free(S2_data);
	free(C2_data);
	free(D1_data);
	free(D2_data);
	free(D3_data);	
	free(D1_weights);
	free(D2_weights);
	free(D3_weights);
	
	cudaFree(S2_datacuda);
	cudaFree(C2_datacuda);
	cudaFree(D1_datacuda);
	cudaFree(D2_datacuda);
	cudaFree(D3_datacuda);		
    cudaFree(raw_datacuda);
    cudaFree(C1_datacuda);
    cudaFree(S1_datacuda);
    cudaFree(C1_kernelcuda);
    cudaFree(Mean_kernelcuda);
    cudaFree(Bias_0cuda);
    cudaFree(Bias_1cuda);
    cudaFree(Bias_2cuda);
    cudaFree(Bias_3cuda);
    cudaFree(Bias_4cuda);
	cudaFree(D1_weightscuda);
	cudaFree(D2_weightscuda);
	cudaFree(D3_weightscuda);
    return 0;
}

	
	
