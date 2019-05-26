#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Kernel128_one.h"
#include "Kernel128_winograd.h"
#include "Kernel256_one.h"
#include "Kernel256_winograd.h"
#include "util.h"

int main(int argc, char** argv) {
  int nTest = 100, sum = 0, sum_cudnn = 0, i;
  cudaSetDevice(0);

  int mode = 0;
  if (argc == 2) {
    mode = atoi(argv[1]);
  }

  for (i = 0; i < nTest; i++) {
    printf("---- Iter: %d ----\n", i);
    int res = -1;
    switch (mode) {
      case 0:
        res = kernel_128();
        break;
      case 1:
        res = kernel_256();
        break;
      case 2:
        res = kernel_128_1_in();
        break;
      case 3:
        res = kernel_128_1_out();
        break;
      case 4:
        res = kernel_256_1_in();
        break;
      case 5:
        res = kernel_256_1_out();
        break;
    }
    if (i > 1) {
      sum += res >> 16;
      sum_cudnn += res & 0xFFFF;
    }
  }
  printf(
      "Average Total Time: [Mine: %d us], [cuDNN: %d us]\n",
      sum / (nTest - 2),
      sum_cudnn / (nTest - 2));

  return 0;
}
