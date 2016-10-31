/*************************************************************************
	> File Name: ./mlp.c
	> Author: zhaowl
	> mail: cryinlaugh@126.com
	> Created Time: Tue 06 Sep 2016 02:51:56 PM CST
 ************************************************************************/

#include "mlp.h"
void mlpForward(ConvData* param){
	Type* input = param->input;
	Type* weight = param->weight;
	Type* bias = param->bias;
	Type* output = param->output;
	int B = param->_B;
	int Ni = param->_Ni;
	int No = param->_No;
	int cB, cNi, cNo;
	for(cB=0; cB<B; cB++){
		for(cNo=0; cNo<No; cNo++){
			output[cNo*B+cB] = bias[cNo]; 
			for(cNi=0; cNi<Ni; cNi++){
				output[cNo*B+cB] += weight[cNo*Ni+cNi]*input[cNi*B+cB];
			}
		}
	}
}
//void mlpForward(ConvData* param, Type* output){
//	Type* input = param->input;
//	Type* weight = param->weight;
//	Type* bias = param->bias;
//	int B = param->_B;
//	int Ni = param->_Ni;
//	int No = param->_No;
//	int cB, cNi, cNo;
//	for(cB=0; cB<B; cB++){
//		for(cNo=0; cNo<No; cNo++){
//			output[cNo*B+cB] = bias[cNo]; 
//			for(cNi=0; cNi<Ni; cNi++){
//				output[cNo*B+cB] += weight[cNo*Ni+cNi]*input[cNi*B+cB];
//			}
//		}
//	}
//}

void swMlpForward(ConvData* param){
	
	char transa = 'n';
	char transb = 'n';
	int B= param->_B;
	int Ni= param->_Ni;
	int No= param->_No;
	Type alpha=1.0;
	Type beta = 1.0;
	int lda=B;
	int ldb=Ni;
	int ldc=B;
	int cB, cNo;
	for(cNo=0; cNo<No; cNo++){
		for(cB=0;cB<B; cB++){
			param->output[cNo*B+cB] = param->bias[cNo];
		}
	}
	//dgemm('n', 'n', 3, 2, 4, 1.0, test_a, 3, test_b, 4, 1.0, test_c, 3);
	dgemm_(&transa, &transb, &B, &No, &Ni, &alpha, param->input, &lda, param->weight, &ldb, &beta, param->output, &ldc);
}

void calcBiasDelta(ConvData* param){
	printf("mlp.c: Enter calcBiasDelta\n");
	int B = param->_B;
	int No = param->_No;
	Type* output_diff = param->output_diff;
	Type* bias_delta = param->bias_delta;
	int cB, cNo;
	//printf("HERE B=%d, No=%d\n", B, No);
	for(cNo=0; cNo<No; cNo++){
		//printf("TEST cNo=%d\n", cNo);
		Type tsum=0;
		for(cB=0; cB<B; cB++){
			tsum+=output_diff[cNo*B+cB];
		}
		bias_delta[cNo] = tsum;
	}
	return;
}
void calcWeightDelta(ConvData* param, Type* weight_delta){
	printf("mlp.c: Enter calcWeightDelta\n");
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	//Type* weight_delta = param->weight_delta;
	Type* output_diff = param->output_diff;
	Type* input = param->input;
	int cB, cNo, cNi;
	//printf("in calcWeightDalta Ni=%d, No=%d, B=%d\n", Ni, No, B);
	for(cNi=0;cNi<Ni;cNi++){
		for(cNo=0;cNo<No;cNo++){
			Type tvalue = 0.0;
			for(cB=0;cB<B;cB++){
				//if(cB==3&&cNi==0) printf("here cNo=%d, input=%f, output_diff=%f\n", cNo, input[cNi*B+cB], output_diff[cNo*B+cB]);	
				tvalue += input[cNi*B+cB]*output_diff[cNo*B+cB];
			}
			weight_delta[cNo*Ni+cNi] = tvalue;
		}
	}
	return;
}
void calcInputDiff(ConvData* param, Type* input_diff){
	printf("mlp.c: Enter calcInputDiff\n");
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	Type* weight = param->weight;
	//Type* input_diff = param->input_diff;
	Type* output_diff = param->output_diff;
	//printf("in calcInputDiff Ni=%d, No=%d, B=%d\n", Ni, No, B);
	int cB, cNo, cNi;
	for(cNi=0; cNi<Ni; cNi++){
		for(cB=0; cB<B; cB++){
			Type tvalue = 0.0;
			for(cNo=0; cNo<No; cNo++){
				tvalue += weight[cNo*Ni+cNi]*output_diff[cNo*B+cB];
				//if(cB==3&&cNi==0) printf("here cNo=%d, weight=%f, output_diff=%f, sum=%.15lf\n", cNo, weight[cNo*Ni+cNi], output_diff[cNo*B+cB], tvalue);
			}
			input_diff[cNi*B+cB] = tvalue;
		}
	}
	return;
}
void mlpBackward(ConvData* param, Type* tweight_delta, Type* tinput_diff){
	//calc bias_delta
	calcBiasDelta(param);
	//calc weight_delta
	calcWeightDelta(param, tweight_delta);
	//calc input_diff
	calcInputDiff(param, tinput_diff);
}
void swCalcWeightDelta(ConvData* param){
	printf("mlp.c: Enter swCalcWeightDelta\n");
	//input=Ni*B
	//output_diff=No*B
	//weight_delta = output_diff*input.trans() = No*Ni
	//In Fortran:
	//weight_delta = Ni*No = (Ni*B)*(B*No) = input.trains()*output_diff
	
	char transa = 't';
	char transb = 'n';
	int B= param->_B;
	int Ni= param->_Ni;
	int No= param->_No;
	//printf("in swCalcWeightDelta Ni=%d, No=%d, B=%d\n", Ni, No, B);
	Type alpha=1.0;
	Type beta = 0.0;
	int lda=B;
	int ldb=B;
	int ldc=Ni;
	//memset(param->weight_delta, 0, Ni*No*sizeof(Type));
	//dgemm('n', 'n', 3, 2, 4, 1.0, test_a, 3, test_b, 4, 1.0, test_c, 3);
	dgemm_(&transa, &transb, &Ni, &No, &B, &alpha, param->input, &lda, param->output_diff, &ldb, &beta, param->weight_delta, &ldc);
}
void swCalcInputDiff(ConvData* param){
	printf("mlp.c: Enter swCalcInputDiff\n");
	//weight = No*Ni
	//output_diff=No*B
	//input_diff = weight.trans()*output_diff  = Ni*B
	//In Fortran:
	//input_diff = B*Ni = output_diff * weight.trans()
	char transa = 'n';
	char transb = 't';
	int B= param->_B;
	int Ni= param->_Ni;
	int No= param->_No;
	//printf("in swCalcInputDiff Ni=%d, No=%d, B=%d\n", Ni, No, B);
	Type alpha=1.0;
	Type beta = 0.0;
	int lda=B;
	int ldb=Ni;
	int ldc=B;
	//memset(param->input_diff, 0, Ni*B*sizeof(Type));
	//dgemm('n', 'n', 3, 2, 4, 1.0, test_a, 3, test_b, 4, 1.0, test_c, 3);
	dgemm_(&transa, &transb, &B, &Ni, &No, &alpha, param->output_diff, &lda, param->weight, &ldb, &beta, param->input_diff, &ldc);
}

void swMlpBackward(ConvData* param){
	//calc bias_delta TODO
	calcBiasDelta(param);
	//calc weight_delta
	swCalcWeightDelta(param);
	//calc input_diff
	swCalcInputDiff(param);
}
