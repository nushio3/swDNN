/*************************************************************************
	> File Name: ./data_def.h
	> Author: zhaowl 
	> mail: cryinlaugh@126.com
	> Created Time: Thu 22 Sep 2016 01:47:46 PM CST
 ************************************************************************/

#ifndef _DATA_DEF_H_
#define _DATA_DEF_H_

#define Type double 
#define SIMDType doublev4
#define THREADS 64 
#define NUM_CG 4 

//Do not change the sequence, hard define for convforward.S
typedef struct ConvData_st{
	//constant def, Do not change!!
	Type* input; //0
	Type* weight; //8
	Type* output; //16
	//   24,  28,  32,  36, 40,  44,  48, 52, 56 
	int _Ni, _Ri, _Ci, _No, _K, _Ro, _Co, _B, _Costride;

	//other defs
	//conv only	
	int _Stride, _Pad, _Biasize;

	Type* bias;
	Type* output_diff;
	Type* input_diff;
	Type* weight_delta;
	Type* bias_delta;

	//softmax
	int* label;
	Type loss;

}ConvData;

typedef struct Network_st{
	int _B;
	int nLayer;
	char** LayerType;
	int** Config;
	Type* InputData;
	Type** LayerData;
	Type** DiffData;
	int* Label;
	Type** Weight;
	Type** WeightDelta;
	Type** Bias;
	Type** BiasDelta;

	long totalMemSize;
	ConvData** Layers;
}NetworkData;

//extern struct ConvData* newConvData();
//extern struct NetworkData* newNetworkData();
#endif
