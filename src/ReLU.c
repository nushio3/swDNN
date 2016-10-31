/*************************************************************************
	> File Name: ./ReLU.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Sat 03 Sep 2016 10:17:01 AM CST
 ************************************************************************/

#include "ReLU.h"


void reluForward(ConvData* param, Type* output){

	Type* input = param->input;
	int B = param->_B;
	int Ni = param->_Ni;
	int No = param->_No;
	if(Ni!=No){
		printf("Wrong input/output size in reluForward!\n");
		exit(0);
	}
	//int Ro = param->_Ro;
	//int Co = param->_Co;
	//int size = B*No*Ro*Co;
	int cN;
	for(cN=0; cN<No; cN++){
		if(input[cN]>0){
			output[cN] = input[cN];
		}else{
			output[cN] = 0;
		}
	}
}

void swReluForward(ConvData* param){
	athread_spawn(reluForward_v1, param);
	athread_join();
}

void reluBackward(ConvData* param, Type* output){
	Type* input = param->input;
	Type* output_diff = param->output_diff;
	int B = param->_B;
	int No = param->_No;
	int Ro = param->_Ro;
	int Co = param->_Co;
	int size = B*No*Ro*Co;
	int cSize;
	for(cSize=0; cSize<size; cSize++){
		if(input[cSize]>0){
			output[cSize] = output_diff[cSize];
		}else{
			output[cSize] = 0;
		}
	}
}
void swReluBackward(ConvData* param){
	athread_spawn(reluBackward_v1, param);
	athread_join();
}
