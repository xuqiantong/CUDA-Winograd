#ifndef __KERNEL256_WINOGRAD_H__
#define __KERNEL256_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName256[] = "data/input_14_1_256.bin";
const char biasName256[] = "data/bias_256.bin";
const char weight_winograd_Name256[] = "data/weight_winograd_256_256.bin";
const char weight_NCHW_Name256[] = "data/weight_NCHW_256_256.bin";

const char bnBiasName256[] = "data/bnBias_256.bin";
const char bnScaleName256[] = "data/bnScale_256.bin";
const char bnBias_winograd_Name256[] = "data/bnBias_winograd_256.bin";
const char bnScale_winograd_Name256[] = "data/bnScale_winograd_256.bin";
const char eMeanName256[] = "data/eMean_256.bin";
const char eVarName256[] = "data/eVar_256.bin";

int kernel_256();

#ifdef __cplusplus
}
#endif

#endif
