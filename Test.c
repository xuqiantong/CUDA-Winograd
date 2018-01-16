#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>

#include "Kernel128_winograd.h"
#include "Kernel256_winograd.h"
#include "Kernel128_one.h"
#include "Kernel256_one.h"
#include "util.h"

int main() {
	int nTest = 100, sum = 0, sum_cudnn = 0, i;
	cudaSetDevice(0); 

	for (i = 0; i < nTest; i++) {
		printf("---- Iter: %d ----\n", i);		
		//int t = kernel_128();
		//int t = kernel_256();
		
		//int t = kernel_128_1_in();
		int t = kernel_128_1_out();
		//int t = kernel_256_1_in();
		//int t = kernel_256_1_out();
		if (i > 1) {
			sum += t >> 16;
			sum_cudnn += t & 0xFFFF;
		}
	}
	printf("Average Total Time: [Mine: %d us], [cuDNN: %d us]\n", sum/(nTest-2), sum_cudnn/(nTest-2));

	return 0;
}

