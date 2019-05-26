#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include "cudnn.h"
#include "util.h"
#include "Kernel128_one.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

__global__ void kernel_512_one_128(float *A, float *B, float *bnBias, float *bnScale, float *C) {
	int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
	int ind = line*128 + in_channel;

	extern __shared__ float shared_[];
	float *weights = shared_ + 512*4, *output = weights + 128*64, *input = shared_;
	float *bias = output + 4*128, *scale = bias + 128;

	for (int i = 0; i < 4; i++)
		input[ind + i*512] = A[tile*2048 + i*512 + ind];
	bias[in_channel] = bnBias[in_channel];
	scale[in_channel] = bnScale[in_channel];
	output[ind] = 0.0f;
	__syncthreads();

	for (int k = 0; k < 512; k += 64) {
		float *B_start = B + k*128;
		for (int i = 0; i < 16; i++)
			weights[ind + i*512] = B_start[i*512 + ind];
		__syncthreads();

		float *A_start = input + k;
		for (int p = 0; p < 64; p++) {
			output[ind] += A_start[line*512 + p] * weights[in_channel + p*128];
		}
		__syncthreads();
	}

	float *C_start = C + tile*512, res = scale[in_channel] * output[ind] + bias[in_channel];
	C_start[ind] = res > 0 ? res : 0;
}


int kernel_128_1_in() {
	float *input = get_parameter(inputName128one, 14*14*512);
	float *weight = get_parameter(weightName128one, 128*512);

	float *bnBias = get_parameter(bnBiasName128one, 128);
	float *bnScale = get_parameter(bnScaleName128one, 128);
	float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 128);
	float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 128);
	float *eMeanName = get_parameter(eMeanName128one, 128);
	float *eVarName = get_parameter(eVarName128one, 128);

	float *input_, *output_, *weight_, *bnBias_, *bnScale_, *eMeanName_, *eVarName_;

	int nInput = 14*14*512, nOutput = 14*14*128, nWeights = 128*512;
	float tmp[nOutput], tmp_cudnn[nOutput];

	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
	cudaError_t s;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////

	/*  1. Data preparation  */
	cudaMalloc((void **) &input_, nInput<<3);
	cudaMalloc((void **) &output_, nOutput<<2);
	cudaMalloc((void **) &weight_, nWeights<<2);
	cudaMalloc((void **) &bnBias_, 128<<2);
	cudaMalloc((void **) &bnScale_, 128<<2);

	cudaMemcpy(input_, input, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnBias_, bnBias_myKernel, 128<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnScale_, bnScale_myKernel, 128<<2, cudaMemcpyHostToDevice);


	/*  2. Computing  */
	nT1 = getTimeMicroseconds64();

	kernel_512_one_128 <<<dim3(49), dim3(128, 4), (4*512 + 64*128 + 4*128 + 2*128)<<2 >>> (input_, weight_, bnBias_, bnScale_, output_);

	//cudaCheckError();
	cudaDeviceSynchronize();

	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1);


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp, output_, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	cudaCheckError();

	free(bnBias_myKernel);
	free(bnScale_myKernel);


	/////////////////////////////////

	// cuDNN

	/////////////////////////////////

	/*  1. Data preparation  */
	cudaMalloc((void **) &eMeanName_, 128<<2);
	cudaMalloc((void **) &eVarName_, 128<<2);

	cudaMemcpy(bnBias_, bnBias, 128<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnScale_, bnScale, 128<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(eMeanName_, eMeanName, 128<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(eVarName_, eVarName, 128<<2, cudaMemcpyHostToDevice);

	weight = transpose(weight, 128, 512);
	cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);

	/*  2. cuDNN preparation  */
	cudnnStatus_t status;
	float one = 1.0, zero = 0.0;
	int size;

	cudnnHandle_t handle;
	status = cudnnCreate(&handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

	cudnnTensorDescriptor_t xdesc, ydesc;
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status = cudnnCreateTensorDescriptor(&xdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
	status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 512, 14, 14);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
	status = cudnnCreateTensorDescriptor(&ydesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
	status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 14, 14);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
	status = cudnnCreateFilterDescriptor(&wdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
	status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 512, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");

	cudnnConvolutionDescriptor_t conv_desc;
	status = cudnnCreateConvolutionDescriptor(&conv_desc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
	status = cudnnSetConvolution2dDescriptor(conv_desc, 0,0, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
	if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");

	cudnnActivationDescriptor_t act_desc;
	status = cudnnCreateActivationDescriptor(&act_desc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");
	status = cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed13\n");

	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	status = cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed14\n");
	status = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed15\n");

	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)0;

	status = cudnnGetConvolutionForwardWorkspaceSize(handle,
	   xdesc,
	   wdesc,
	   conv_desc,
	   ydesc,
	   algo,
	   (size_t *)&(size));
	float *extra;
	cudaMalloc((void **) &extra, size);


	/*  3. Computing  */
	nT1_cudnn = getTimeMicroseconds64();

	status = cudnnConvolutionForward(handle, &one,
		xdesc, input_, wdesc, weight_,
		conv_desc, algo,
		extra, size, &zero,
		ydesc, output_);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed1\n");

	status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
		&one, &zero,
		ydesc, output_, ydesc, output_,
		bnScaleBiasMeanVarDesc, bnScale_, bnBias_, eMeanName_, eVarName_, CUDNN_BN_MIN_EPSILON);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed2\n");

	status = cudnnActivationForward(handle, act_desc, &one,
		ydesc, output_, &zero,
		ydesc, output_);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed3\n");

	cudaDeviceSynchronize();
	nT2_cudnn = getTimeMicroseconds64();
	printf("cuDNN TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);


	/*  4. Copy back and free  */
	s = cudaMemcpy(tmp_cudnn, output_, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));

	cudaFree(extra);
	cudaFree(input_);
	cudaFree(output_);
	cudaFree(weight_);

	cudaFree(bnScale_);
	cudaFree(bnBias_);
	cudaFree(eMeanName_);
	cudaFree(eVarName_);

	free(input);
	free(weight);

	free(bnScale);
	free(bnBias);
	free(eMeanName);
	free(eVarName);

	output_checker(tmp, tmp_cudnn, 14, 128, 0);

	return ((nT2-nT1) << 16) | (nT2_cudnn-nT1_cudnn);
}



__global__ void kernel_128_one_512(float *A, float *B, float *bnBias, float *bnScale, float *C) {
	int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
	int ind = line*128 + in_channel;

	extern __shared__ float shared_[];
	float *weights = shared_ + 128*4, *output = weights + 128*64, *input = shared_;
	float *bias = output + 4*128, *scale = bias + 128;

	input[ind] = A[tile * 512 + ind];
	bias[in_channel] = bnBias[part*128 + in_channel];
	scale[in_channel] = bnScale[part*128+ in_channel];
	output[ind] = 0.0f;
	__syncthreads();

	for (int k = 0; k < 128; k += 64) {
		for (int i = 0; i < 16; i++)
			weights[ind + 512*i] = B[(k + i*4 + line)*512 + part*128 + in_channel];
		__syncthreads();

		float *A_start = input + k;
		for (int p = 0; p < 64; p++) {
			output[ind] += A_start[line*128 + p] * weights[in_channel + p*128];
		}
		__syncthreads();
	}

	float *C_start = C + tile*2048 + part*128;
	float res = scale[in_channel] * output[ind] + bias[in_channel];
	C_start[line * 512 + in_channel] = res;
}


int kernel_128_1_out() {
	float *input = get_parameter(inputName128one, 14*14*128);
	float *weight = get_parameter(weightName128one, 128*512);

	float *bnBias = get_parameter(bnBiasName128one, 512);
	float *bnScale = get_parameter(bnScaleName128one, 512);
	float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 512);
	float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 512);
	float *eMeanName = get_parameter(eMeanName128one, 512);
	float *eVarName = get_parameter(eVarName128one, 512);

	float *input_, *output_, *weight_, *bnBias_, *bnScale_, *eMeanName_, *eVarName_;

	int nInput = 14*14*128, nOutput = 14*14*512, nWeights = 128*512;
	float tmp[nOutput], tmp_cudnn[nOutput];

	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
	cudaError_t s;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////

	/*  1. Data preparation  */
	cudaMalloc((void **) &input_, nInput<<3);
	cudaMalloc((void **) &output_, nOutput<<2);
	cudaMalloc((void **) &weight_, nWeights<<2);
	cudaMalloc((void **) &bnBias_, 512<<2);
	cudaMalloc((void **) &bnScale_, 512<<2);

	cudaMemcpy(input_, input, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnBias_, bnBias_myKernel, 512<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnScale_, bnScale_myKernel, 512<<2, cudaMemcpyHostToDevice);

	/*  2. Computing  */
	nT1 = getTimeMicroseconds64();

	kernel_128_one_512 <<<dim3(49, 4), dim3(128, 4), (4*128 + 64*128 + 4*128 + 2*128)<<2 >>> (input_, weight_, bnBias_, bnScale_, output_);

	//cudaCheckError();
	cudaDeviceSynchronize();

	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1);


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp, output_, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	cudaCheckError();

	free(bnBias_myKernel);
	free(bnScale_myKernel);


	/////////////////////////////////

	// cuDNN

	/////////////////////////////////

	/*  1. Data preparation  */
	cudaMalloc((void **) &eMeanName_, 512<<2);
	cudaMalloc((void **) &eVarName_, 512<<2);

	cudaMemcpy(bnBias_, bnBias, 512<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnScale_, bnScale, 512<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(eMeanName_, eMeanName, 512<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(eVarName_, eVarName, 512<<2, cudaMemcpyHostToDevice);

	weight = transpose(weight, 512, 128);
	cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);

	/*  2. cuDNN preparation  */
	cudnnStatus_t status;
	float one = 1.0, zero = 0.0;
	int size;

	cudnnHandle_t handle;
	status = cudnnCreate(&handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

	cudnnTensorDescriptor_t xdesc, ydesc;
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status = cudnnCreateTensorDescriptor(&xdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
	status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 14, 14);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
	status = cudnnCreateTensorDescriptor(&ydesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
	status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 512, 14, 14);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
	status = cudnnCreateFilterDescriptor(&wdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
	status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 512, 128, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");

	cudnnConvolutionDescriptor_t conv_desc;
	status = cudnnCreateConvolutionDescriptor(&conv_desc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
	status = cudnnSetConvolution2dDescriptor(conv_desc, 0,0, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
	if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");

	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	status = cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed14\n");
	status = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 512, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed15\n");

	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)0;

	status = cudnnGetConvolutionForwardWorkspaceSize(handle,
	   xdesc,
	   wdesc,
	   conv_desc,
	   ydesc,
	   algo,
	   (size_t *)&(size));
	float *extra;
	cudaMalloc((void **) &extra, size);


	/*  3. Computing  */
	nT1_cudnn = getTimeMicroseconds64();

	status = cudnnConvolutionForward(handle, &one,
		xdesc, input_, wdesc, weight_,
		conv_desc, algo,
		extra, size, &zero,
		ydesc, output_);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed1\n");

	status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
		&one, &zero,
		ydesc, output_, ydesc, output_,
		bnScaleBiasMeanVarDesc, bnScale_, bnBias_, eMeanName_, eVarName_, CUDNN_BN_MIN_EPSILON);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed2\n");

	cudaDeviceSynchronize();
	nT2_cudnn = getTimeMicroseconds64();
	printf("cuDNN TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);


	/*  4. Copy back and free  */
	s = cudaMemcpy(tmp_cudnn, output_, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));

	cudaFree(extra);
	cudaFree(input_);
	cudaFree(output_);
	cudaFree(weight_);

	cudaFree(bnScale_);
	cudaFree(bnBias_);
	cudaFree(eMeanName_);
	cudaFree(eVarName_);

	free(input);
	free(weight);

	free(bnScale);
	free(bnBias);
	free(eMeanName);
	free(eVarName);

	output_checker(tmp, tmp_cudnn, 14, 512, 0);

	return ((nT2-nT1) << 16) | (nT2_cudnn-nT1_cudnn);
}
