#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include "def.h"
#include "ConvForward.h"
#include "mlp.h"
#include "softmax.h"
//#include "core_functions.h"

#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))
#define TEST_STEPS 1 

//#define DEBUG
//#define SWONLY 

int init(ConvData* param, Type* test_output);
int check(ConvData* param, Type* coutput, int mode);
void softmax_check(ConvData* param);

int main(int argc, char **argv)
{
	int Ro = 9;
	int Co = 9;
	int Ni = 48; 
	int Ri = 9;
	int Ci = 9;
	int No = 96;
	int K = 3;
	int B = 128;
	int Stride = 1;
	int Np = ((Ro-1)*Stride+K-Ri)/2;
	//int Np = (Ro+K-1-Ri)/2;
	printf("B=%d, Ni=%d, No=%d, Ri=%d, Ci=%d, Ro=%d, Co=%d, K=%d\n",B, Ni, No, Ri, Ci, Ro, Co, K);
	if((Co+K-1-Ci)!=2*Np){
		printf("Wrong padding size. Exit!\n");
		return 0;
	}
	int Costride = (64*56*1024/8-Ni*B*2-Ni*No*K)/(No*B)-(K-1);
	printf("Costride is %d\n", Costride);
	int ldm_consume = 8*(Ni*No*K + No*B*(Costride+K-1) + Ni*B*2); //total ldm consumption in all slave nodes
	printf("ldm comsumption is %d\n", ldm_consume/64);
	assert(ldm_consume < 64*1024*64);

	int s, ii;
	struct timeval t1, t2;	
	struct timeval t3, t4;	
	float gflop = 2.0*(No*Ro*Co*1.0/1000)*(K*K*Ni*1.0/1000)*B*1.0/1000;
	
	double t = 0;
	long int data_size = sizeof(Type)*(B*Ni*Ri*Ci+No*Ni*K*K+2*B*No*Ro*Co);
	printf("Begin allocate %d MByte Mem\n", data_size/1000000);
	Type* input = (Type*) malloc(B*Ni*Ri*Ci*sizeof(Type));
	Type* weight = (Type*) malloc(No*Ni*K*K*sizeof(Type));
	Type* bias = (Type*) malloc(No*sizeof(Type));
	Type* output = (Type*) malloc(B*No*Ro*Co*sizeof(Type));
	Type* test_output = (Type*) malloc(B*No*Ro*Co*sizeof(Type));
	printf("MPE allocate Memory Seccuess!\n");

	//init para struct
	ConvData* param = malloc(sizeof(ConvData));
	param->input = input; //
	param->weight = weight;
	param->output = output;
	param->bias = bias;
	param->_Ni = Ni;
	param->_Ri = Ri;
	param->_Ci = Ci;
	param->_No = No;
	param->_K = K;
	param->_Ro = Ro;
	param->_Co = Co;
	param->_B = B;
	param->_Costride = Costride;
	param->_Stride = Stride;
	param->_Pad = Np;
	printf("Init Data\n");
	
	init(param, test_output);
	
	int it, jt;
#ifdef DEBUG
	printf("Before padding. (%d,%d)\n", param->_Ri, param->_Ci);
	for(it=0; it<param->_Ri; it++){
		for(jt=0; jt<param->_Ci; jt++){
			printf("%f\t", *(param->input+it*param->_Ci*Ni*B+jt*Ni*B));
		}
		printf("\n");
	}
#endif
	printf("Begin Padding.\n");
	gettimeofday(&t3, NULL);
	swZeroPad(param);
	gettimeofday(&t4, NULL);
	printf("Finish Padding. Time: %0.9lfs \n", TIME(t3, t4)); 
#ifdef DEBUG
	printf("After padding. (%d, %d)\n", param->_Ri, param->_Ci);
	for(it=0; it<param->_Ri; it++){
		for(jt=0; jt<param->_Ci; jt++){
			printf("%f\t", *(param->input+it*param->_Ci*Ni*B+jt*Ni*B));
		}
		printf("\n");
	}
#endif

	//call swConvForward()
	//printf("!!!!test %f\n", test_output[127]);
	printf("Begin swConvForward.\n");
	gettimeofday(&t3, NULL);
	swConvForward(param);
	gettimeofday(&t4, NULL);
	printf("Finish swConvForward. Time: %0.9lfs FLOPS:%.5f GFLOPS\n", TIME(t3, t4), gflop*STEPS/(TIME(t3, t4))); 
	
	//call conv on host (for validation) and check results
	//printf("!!!!test %f\n", test_output[127]);
	printf("Begin conv on cpu. \n");
	gettimeofday(&t1, NULL);
	//CaffeConv(param->input, param->weight, test_output);
	CaffeConv(param, test_output);
	gettimeofday(&t2, NULL);
	printf("Finish conv on cpu. Time: %0.9lfs FLOPS:%.5f GFLOPS\n", TIME(t1, t2), gflop/(TIME(t1, t2)));
	//CaffeConv2(input, weight, output);
	printf("#### check convforward results.\n");
	check(param, test_output, 0);

#ifdef DEBUG
	printf("Before add bias. (%d,%d)\n", param->_Ro, param->_Co);
	for(it=0; it<param->_Ro; it++){
		for(jt=0; jt<param->_Co; jt++){
			printf("%f\t", *(param->output+it*param->_Co*No*B+jt*No*B));
		}
		printf("\n");
	}
#endif
	
	printf("Begin convAddBias on cpu.\n");
	gettimeofday(&t3, NULL);
	convAddBias(param, test_output);
	gettimeofday(&t4, NULL);
	printf("Finish convAddBias on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("Begin swAddBias.\n");
	gettimeofday(&t3, NULL);
	swConvAddBias(param);
	gettimeofday(&t4, NULL);
	printf("Finish swAddBias. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("#### check addBias results.\n");
	check(param, test_output, 0);

#ifdef DEBUG
	printf("After add bias. (%d, %d) bias=%f\n", param->_Ro, param->_Co,param->bias[0]);
	for(it=0; it<param->_Ro; it++){
		for(jt=0; jt<param->_Co; jt++){
			printf("%f\t", *(param->output+it*param->_Co*No*B+jt*No*B));
			//printf("%f\t", *(test_output+it*param->_Co*No*B+jt*No*B));
		}
		printf("\n");
	}
#endif

#ifdef DEBUG
	printf("Before relu. (%d,%d)\n", param->_Ro, param->_Co);
	for(it=0; it<param->_Ro; it++){
		for(jt=0; jt<param->_Co; jt++){
			printf("%f\t", *(param->output+it*param->_Co*No*B+jt*No*B));
		}
		printf("\n");
	}
#endif

	printf("Begin Relu on cpu.\n");
	gettimeofday(&t3, NULL);
	reluForward(param, test_output);
	gettimeofday(&t4, NULL);
	printf("Finish Relu on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("Begin swRelu.\n");
	gettimeofday(&t3, NULL);
	swReluForward(param);
	gettimeofday(&t4, NULL);
	printf("Finish swRelu. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("#### check reluforwad results.\n");
	check(param, test_output, 0);

#ifdef DEBUG
	printf("After relu. (%d, %d)\n", param->_Ro, param->_Co);
	for(it=0; it<param->_Ro; it++){
		for(jt=0; jt<param->_Co; jt++){
			printf("%f\t", *(param->output+it*param->_Co*No*B+jt*No*B));
		}
		printf("\n");
	}
#endif

	//define MLP configerations
	int mNi = No*Ro*Co;
	int mNo = Ro*Co;
	ConvData* mlp_param = malloc(sizeof(ConvData));
	Type* mlp_weight = malloc(sizeof(Type)*mNi*mNo);
	Type* mlp_output = malloc(sizeof(Type)*B*mNo);
	Type* mlp_bias = malloc(sizeof(Type)*mNo);
	mlp_param->input = param->output;
	mlp_param->weight = mlp_weight;
	mlp_param->output = mlp_output;
	mlp_param->bias = mlp_bias;
	mlp_param->_Ni = mNi;
	mlp_param->_No = mNo;
	mlp_param->_B  = B;
	mlp_init(mlp_param);
	Type* tmlp_output = malloc(sizeof(Type)*B*mNo);
	for(it=0; it<B*mNo; it++){
		tmlp_output[it] = mlp_param->output[it];	
	}
	printf("Begin mlp with dgemm:\n");
	gettimeofday(&t3, NULL);
	swMlpForward(mlp_param);
	gettimeofday(&t4, NULL);
	printf("Finish mlp with dgemm. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("Begin mlp on cpu:\n");
	gettimeofday(&t3, NULL);
	mlpForward(mlp_param, tmlp_output);
	gettimeofday(&t4, NULL);
	printf("Finish mlp on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 

	ConvData* softmax_param = (ConvData*) malloc(sizeof(ConvData));
	softmax_param->input = mlp_param->output;
	softmax_param->_B = mlp_param->_B;
	softmax_param->_Ni = mlp_param->_No;
	softmax_param->_No = softmax_param->_Ni;
	Type* softmax_output = (Type*) malloc(sizeof(Type)*softmax_param->_No*softmax_param->_B);
	softmax_param->output = softmax_output;
	int* label = (int*) malloc(softmax_param->_B*softmax_param->_No*sizeof(int));
	softmax_param->label = label;
	label_init(softmax_param);
	Type* softmax_diff = (Type*) malloc(softmax_param->_B*softmax_param->_No*sizeof(Type));
	softmax_param->input_diff = softmax_diff;
	
#ifdef DEBUG
	printf("Before softmax input:");
	for(it=0; it<Ro; it++){
		for(jt=0; jt<Co; jt++){
			printf("%.10lf\t", softmax_param->input[(it*Co+jt)*B+3]);
		}
		printf("\n");
	}
#endif
	printf("Begin softmax on cpu:\n");
	gettimeofday(&t3, NULL);
	softmaxForward(softmax_param);
	gettimeofday(&t4, NULL);
	printf("Finish softmax on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("#### Check softmax results: \n");
	softmax_check(softmax_param);


#ifdef DEBUG
	printf("After softmax input:");
	for(it=0; it<Ro; it++){
		for(jt=0; jt<Co; jt++){
			printf("%.10lf\t", softmax_param->input[(it*Co+jt)*B+3]);
		}
		printf("\n");
	}
	printf("After softmax output:");
	for(it=0; it<Ro; it++){
		for(jt=0; jt<Co; jt++){
			printf("%.10lf\t", softmax_param->output[(it*Co+jt)*B+3]);
		}
		printf("\n");
	}
#endif
	
#ifdef DEBUG	
	printf("Before calc loss:");
	for(it=0; it<Ro; it++){
		for(jt=0; jt<Co; jt++){
			printf("%.10lf\t", softmax_param->output[(it*Co+jt)*B+3]);
		}
		printf("\n");
	}
#endif
	
	printf("Calc Loss:\n");
	softmaxWithLoss(softmax_param);
	printf("Done calc loss = %lf\n", softmax_param->loss);
	
	printf("Begin Backpropagation!\n");
	
	printf("Begin softmax_bp on cpu:\n");
	gettimeofday(&t3, NULL);
	softmaxBackward(softmax_param);
	gettimeofday(&t4, NULL);
	printf("Finish softmax_bp on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 

#ifdef DEBUG
	printf("After softmax_bp input_diff:");
	for(it=0; it<Ro; it++){
		for(jt=0; jt<Co; jt++){
			printf("%f\t", softmax_param->input_diff[(it*Co+jt)*B+3]);
		}
		printf("\n");
	}
#endif

	Type* mlp_weight_delta = (Type*) malloc(mlp_param->_Ni*mlp_param->_No*sizeof(Type));
	Type* mlp_bias_delta = (Type*) malloc(mlp_param->_No*sizeof(Type));
	Type* mlp_input_diff = (Type*) malloc(mlp_param->_Ni*mlp_param->_B*sizeof(Type));
	Type* mlp_tweight_delta = (Type*) malloc(mlp_param->_Ni*mlp_param->_No*sizeof(Type));
	Type* mlp_tinput_diff = (Type*) malloc(mlp_param->_Ni*mlp_param->_B*sizeof(Type));
	mlp_param->weight_delta = mlp_weight_delta;
	mlp_param->bias_delta = mlp_bias_delta;
	mlp_param->input_diff = mlp_input_diff;
	mlp_param->output_diff = softmax_param->input_diff;

	printf("Begin mlp_bp with dgemm:\n");
	gettimeofday(&t3, NULL);
	swMlpBackward(mlp_param);
	gettimeofday(&t4, NULL);
	printf("Finish mlp_bp with dgemm. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("Begin mlp_bp on cpu:\n");
	gettimeofday(&t3, NULL);
	mlpBackward(mlp_param, mlp_tweight_delta, mlp_tinput_diff);
	gettimeofday(&t4, NULL);
	printf("Finish mlp_bp on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("#### check mlp_bp bias_delta results.\n");
	bias_delta_check(mlp_param);
	printf("#### check mlp_bp weight_delta results.\n");
	weight_delta_check(mlp_param, mlp_tweight_delta);
	printf("#### check mlp_bp input_diff results.\n");
	input_diff_check(mlp_param, mlp_tinput_diff);

	param->output_diff = mlp_param->input_diff;
	printf("Begin swRelu_bp.\n");
	gettimeofday(&t3, NULL);
	swReluBackward(param);
	gettimeofday(&t4, NULL);
	printf("Finish swRelu_bp. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("Begin Relu_bp on cpu.\n");
	gettimeofday(&t3, NULL);
	reluBackward(param, test_output);
	gettimeofday(&t4, NULL);
	printf("Finish Relu_bp on cpu. Time: %0.9lfs \n", TIME(t3, t4)); 
	printf("#### check relubackward results.\n");
	check(param, test_output, 1);
	
	printf("begin free!\n");
	free(input);
	free(weight);
	free(bias);
	free(output);
	free(test_output);
	free(mlp_weight);
	free(mlp_output);
	free(mlp_bias);
	free(tmlp_output);
	free(softmax_output);
	free(label);
	free(softmax_diff);
	free(mlp_weight_delta);
	free(mlp_tweight_delta);
	free(mlp_bias_delta);
	free(mlp_input_diff);
	free(mlp_tinput_diff);
	printf("end program!\n");
	return 0;

}
int label_init(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	int* label = param->label;
	int cB, cNo;
	//memset(label, 0, B*No*sizeof(int));
	for(cB=0; cB<B; cB++){
		for(cNo=0; cNo<No; cNo++){
			if(cNo==3) label[cNo*B+cB]=1;
			else label[cNo*B+cB]=0;
		}
	}
}
int mlp_init(ConvData* param){
	Type* weight = param->weight;
	Type* output = param->output;
	Type* bias = param->bias;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;

	int i,j;
	srand((unsigned int) time (NULL));  
	for(i=0; i < Ni*No; i++){
		//input[i] = i%100*0.01; //rand()*1.0/RAND_MAX;
		weight[i] = rand()*0.01/RAND_MAX;
	}
	for(i=0; i < No; i++){
		//input[i] = i%100*0.01; //rand()*1.0/RAND_MAX;
		bias[i] = (i+1)*0.01;
	}
}
//int init(Type* input, Type* weight, Type* output, Type* test_output)
int init(ConvData* param, Type* test_output)
{
	Type* input = param->input;
	Type* weight = param->weight;
	Type* bias = param->bias;
	Type* output = param->output;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	int Ro = param->_Ro;
	int Co = param->_Co;
	int Ri = param->_Ri;
	int Ci = param->_Ci;
	int K = param->_K;
	
	int i;
	srand((unsigned int) time (NULL));  
	for(i=0; i < B*Ni*Ri*Ci; i++){
		//input[i] = i%100*0.01; //rand()*1.0/RAND_MAX;
		input[i] = rand()*0.01/RAND_MAX;
	}
	int sign=-1;
	for(i=0; i< Ni*No*K*K; i++){
		//weight[i] = i%100 * 0.01; //rand()*1.0/RAND_MAX;
		weight[i] = sign*rand()*0.01/RAND_MAX;
		sign*=-1;
	}
	for(i=0; i<No; i++){
		bias[i] = (i+1)*0.01;
	}
	for(i=0; i< B*No*Ro*Co; i++){
		output[i] = 0.0;
		test_output[i] = 0.0;
	}
	return 0;
}
int check(ConvData* param, Type* coutput, int mode){
	Type* input = param->input;
	Type* weight = param->weight;
	Type* output;
	if(mode==0) output = param->output;
	else output = param->output_diff;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	int Ro = param->_Ro;
	int Co = param->_Co;
	int K = param->_K;
	
	int i;
	int n=0;
	for(i=0; i<B*No*Ro*Co; i++){
		Type diff = *(output+i)-*(coutput+i);
		if(diff>1e-10||diff<-1e-10){
			printf("Wrong at %d, %.15lf, %.15lf\n", i, *(output+i), *(coutput+i));
			n++;
			if(n>64) break;
		}
	}
	Type sum1 = 0.0, sum2 = 0.0;
	for(i=0; i<B*No*Ro*Co; i++){
		sum1 += *(output+i);
		sum2 += *(coutput+i);
	}
	printf("sum1 %lf sum2 %lf\n", sum1, sum2);
}
int bias_delta_check(ConvData* param){
	Type sum=0.0;
	int cNo;
	for(cNo=0;cNo<param->_No; cNo++){
		sum+=param->bias_delta[cNo];
		//printf("%f\t", param->bias_delta[cNo]);
	}
	printf("sum=%f\n", sum);
}
int input_diff_check(ConvData* param, Type* coutput){
	Type* output = param->input_diff;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	int i;
	int n=0;
	for(i=0; i<Ni*B; i++){
		Type diff = *(output+i)-*(coutput+i);
		if(diff>1e-8||diff<-1e-8){
			printf("Wrong at %d, %.15lf, %.15lf\n", i, *(output+i), *(coutput+i));
			n++;
			if(n>64) break;
		}
	}
	Type sum1 = 0.0, sum2 = 0.0;
	for(i=0; i<Ni*B; i++){
		sum1 += *(output+i);
		sum2 += *(coutput+i);
	}
	printf("sum1 %lf sum2 %lf\n", sum1, sum2);
}
int weight_delta_check(ConvData* param, Type* coutput){
	Type* output = param->weight_delta;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	
	int i;
	int n=0;
	for(i=0; i<Ni*No; i++){
		Type diff = *(output+i)-*(coutput+i);
		if(diff>1e-8||diff<-1e-8){
			printf("Wrong at %d, %.15lf, %.15lf\n", i, *(output+i), *(coutput+i));
			n++;
			if(n>64) break;
		}
	}
	Type sum1 = 0.0, sum2 = 0.0;
	for(i=0; i<Ni*No; i++){
		sum1 += *(output+i);
		sum2 += *(coutput+i);
	}
	printf("sum1 %lf sum2 %lf\n", sum1, sum2);
}
int mlp_check(ConvData* param, Type* coutput){
	Type* input = param->input;
	Type* weight = param->weight;
	Type* output = param->output;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	int Ro = param->_Ro;
	int Co = param->_Co;
	int K = param->_K;
	
	int i;
	int n=0;
	for(i=0; i<B*No; i++){
		Type diff = *(output+i)-*(coutput+i);
		if(diff>1e-8||diff<-1e-8){
			printf("Wrong at %d, %.15lf, %.15lf\n", i, *(output+i), *(coutput+i));
			n++;
			if(n>64) break;
		}
	}
	Type sum1 = 0.0, sum2 = 0.0;
	for(i=0; i<B*No; i++){
		sum1 += *(output+i);
		sum2 += *(coutput+i);
	}
	printf("sum1 %lf sum2 %lf\n", sum1, sum2);
}
void softmax_check(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	Type* output = param->output;
	int cB,  cNo;
	for(cB=0; cB<B; cB++){
		Type sum=0;
		for(cNo=0; cNo<No; cNo++){
			sum+=output[cNo*B+cB];
		}
		if((sum-1)>1e-9 || (sum-1)<-1e-9){
		//if(sum!=1){
			printf("Wrong at batch num %d, sum=%f\n", cB, sum);
			for(cNo=0; cNo<No; cNo++){
				printf("%f\t", output[cNo*B+cB]);
			}
			printf("\n");
			return ;
		}
	}
	printf("Check passed!\n");
}
