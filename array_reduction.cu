#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 512 // You can change this
#define NUM_OF_ELEMS 1048576 // You can change this

/* #define funcCheck(stmt) {                                            \ */
/*     cudaError_t err = stmt;                                          \ */
/*     if (err != cudaSuccess)                                          \ */
/*     {                                                                \ */
/*         printf( "Failed to run stmt %d ", __LINE__);                 \ */
/*         printf( "Got CUDA error ...  %s ", cudaGetErrorString(err)); \ */
/*         return -1;                                                   \ */
/*     }                                                                \ */
/* } */

__global__  void array_sum(float * input, float * output, int len){
	// Load a segment of the input vector into shared memory
	__shared__ float partialSum[2*BLOCK_SIZE];
	int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;

	// load data to shared memory
	// load first half
	if ((start + t) < len){
		partialSum[t] = input[start + t];
	}
	else{
		partialSum[t] = 0.0;
	}

	// load latter half
	if ((start + blockDim.x + t) < len){
		partialSum[blockDim.x + t] = input[start + blockDim.x + t];
	}
	else{
		partialSum[blockDim.x + t] = 0.0;
	}

	__syncthreads();

	// Traverse reduction tree
	// start to add
	for (int stride = blockDim.x; stride > 0; stride /= 2){
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
		__syncthreads();
	}

	// Write the computed sum of the block to the output vector at correct index
	if (t == 0 && (globalThreadId*2) < len){
		output[blockIdx.x] = partialSum[t];
	}
}

int main()
{
	// host data
	float * h_input; // The input 1D vector

	// host output
	float * h_output; // The output vector

	//device device data
	float * d_input;

	// device output data
	float * d_output;

	int numOutputElements; // number of elements in the output list
	h_input = (float*)malloc(sizeof(float) * NUM_OF_ELEMS);

	// allocate host array
	for (int i=0; i < NUM_OF_ELEMS; i++){
		h_input[i] = 1.0;     // Add your input values
	}

	//int NUM_OF_ELEMS = NUM_OF_ELEMS; // number of elements in the input list
	numOutputElements = NUM_OF_ELEMS / (BLOCK_SIZE*2);
	if (NUM_OF_ELEMS % (BLOCK_SIZE*2)){
		numOutputElements++;
	}
	h_output = (float*) malloc(numOutputElements * sizeof(float));

	//@@ Allocate GPU memory here
	cudaMalloc((void **)&d_input, NUM_OF_ELEMS * sizeof(float));
	cudaMalloc((void **)&d_output, numOutputElements * sizeof(float));

	// Copy memory to the GPU here
	cudaMemcpy(d_input, h_input, NUM_OF_ELEMS * sizeof(float), cudaMemcpyHostToDevice);

	// Initialize the grid and block dimensions here
	dim3 DimGrid( numOutputElements, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);

	// Launch the GPU Kernel here
	array_sum<<<DimGrid, DimBlock>>>(d_input, d_output, NUM_OF_ELEMS);

	// Copy the GPU memory back to the CPU here
	cudaMemcpy(h_output, d_output, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

	/********************************************************************
	 * Reduce output vector on the host
	 ********************************************************************/
	for (int j = 1; j < numOutputElements; j++){
		h_output[0] += h_output[j];
	}
	printf("Reduced Sum from GPU = %f\n", h_output[0]);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to cuda reduction %f ms\n", milliseconds);

	// Free the GPU memory here
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);

	return 0;
}
