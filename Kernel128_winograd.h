#ifndef __KERNEL128_WINOGRAD_H__
#define __KERNEL128_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName128[] = "data/input_14_1_128.bin";
const char biasName128[] = "data/bias_128.bin";
const char weight_winograd_Name128[] = "data/weight_winograd_128_128.bin";
const char weight_NCHW_Name128[] = "data/weight_NCHW_128_128.bin";

const char bnBiasName128[] = "data/bnBias_128.bin";
const char bnScaleName128[] = "data/bnScale_128.bin";
const char bnBias_winograd_Name128[] = "data/bnBias_winograd_128.bin";
const char bnScale_winograd_Name128[] = "data/bnScale_winograd_128.bin";
const char eMeanName128[] = "data/eMean_128.bin";
const char eVarName128[] = "data/eVar_128.bin";

int kernel_128();

#ifdef __cplusplus
}
#endif

#endif