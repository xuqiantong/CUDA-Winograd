#ifndef __KERNEL128_ONE_H__
#define __KERNEL128_ONE_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName128one[] = "data/input_one_14_1024.bin";
const char weightName128one[] = "data/weight_one_1024.bin";

const char bnBiasName128one[] = "data/bnBias_one_1024.bin";
const char bnScaleName128one[] = "data/bnScale_one_1024.bin";
const char bnBias_myKernel_Name128one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name128one[] = "data/bnScale_myKernel_one_1024.bin";
const char eMeanName128one[] = "data/eMean_one_1024.bin";
const char eVarName128one[] = "data/eVar_one_1024.bin";

int kernel_128_1_in();
int kernel_128_1_out();

#ifdef __cplusplus
}
#endif

#endif