/*************************************************************************
	> File Name: ./softmax.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Wed 07 Sep 2016 03:48:38 PM CST
 ************************************************************************/

#include "softmax.h"

void softmaxForward(ConvData* param){
	Type* input = param->input;
	Type* output = param->output;
	int B = param->_B;
	int Ni = param->_Ni;
	int No = param->_No;
	int cB, cNi, cNo;
	
	//substruct max to avoid overflow of exp() {caffe impl.}
	for(cB=0; cB<B; cB++){
		Type tmax = input[cB];
		for(cNi=1; cNi<Ni; cNi++){
			if(tmax < input[cNi*B+cB]) tmax = input[cNi*B+cB];
		}
		Type tsum_exp = 0;
		for(cNi=0; cNi<Ni; cNi++){
			input[cNi*B+cB] = input[cNi*B+cB]-tmax;
			output[cNi*B+cB] = exp(input[cNi*B+cB]);
			//output[cNi*B+cB] = input[cNi*B+cB];
			tsum_exp += output[cNi*B+cB];
		}
		for(cNi=0; cNi<Ni; cNi++){
			output[cNi*B+cB] = output[cNi*B+cB]/tsum_exp;
		}
	}
}
void softmaxWithLoss(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	Type* output = param->output;
	int* label = param->label;
	int cB, cNo;
	Type sum_log_likelihood=0;
	for(cB=0; cB<B; cB++){
		for(cNo=0; cNo<No; cNo++){
			if(label[cNo*B+cB]==1){
				//printf("here B = %d %.15lf, %.15lf\n", cB,output[cNo*B+cB],sum_log_likelihood);
				sum_log_likelihood += log(output[cNo*B+cB]);
			}else if(label[cNo*B+cB]!=0){
				printf("WRONG LABEL!!!! cB = %d, cNo = %d, label = %lf\n", cB, cNo, label[cNo*B+cB]);
				return;
			}
		}
	}
	param->loss = -sum_log_likelihood/B;
}

void softmaxBackward(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	int* label = param->label;
	Type* output = param->output;
	Type* input_diff = param->input_diff;
	int cB, cNo;
	for(cB=0; cB<B; cB++){
		for(cNo=0; cNo<No; cNo++){
			input_diff[cNo*B+cB] = output[cNo*B+cB]-label[cNo*B+cB];
		}
	}
}
