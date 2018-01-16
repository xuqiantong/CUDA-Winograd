#ifndef __KERNEL256_ONE_H__
#define __KERNEL256_ONE_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName256one[] = "data/input_one_14_1024.bin";
const char weightName256one[] = "data/weight_one_1024.bin";

const char bnBiasName256one[] = "data/bnBias_one_1024.bin";
const char bnScaleName256one[] = "data/bnScale_one_1024.bin";
const char bnBias_myKernel_Name256one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name256one[] = "data/bnScale_myKernel_one_1024.bin";
const char eMeanName256one[] = "data/eMean_one_1024.bin";
const char eVarName256one[] = "data/eVar_one_1024.bin";

int kernel_256_1_in();
int kernel_256_1_out();

#ifdef __cplusplus
}
#endif

#endif