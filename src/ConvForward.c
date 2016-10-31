/*************************************************************************
	> File Name: ./ConvForward.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Sat 03 Sep 2016 03:30:14 PM CST
 ************************************************************************/

#include "ConvForward.h"

void CaffeConv(ConvData* param){
	Type* input = param->input;
	Type* output = param->output;
	Type* weight = param->weight;
	int B = param->_B;
	int No = param->_No;
	int Ni = param->_Ni;
	int Ro = param->_Ro;
	int Co = param->_Co;
	int K = param->_K;

	int cB,cNo,cNi,cRo,cCo,cKr,cKc;
	printf("begin for ... B=%d, Ni=%d, No=%d, Ro=%d, Co=%d, K=%d\n", B, Ni, No, Ro, Co, K);
	for(cRo=0; cRo<Ro; cRo++)
		for(cCo=0; cCo<Co; cCo++){

			for(cNo=0; cNo<No; cNo++){

				for(cKr = 0 ;cKr<K; cKr++)
					for(cKc = 0; cKc<K; cKc++){
						for(cNi = 0; cNi<Ni; cNi++){
							for(cB = 0; cB<B; cB++){
								//if(cRo==0&&cCo==18&&cB==0) printf("HERE (cRo=%d, cCo=%d, cNo=%d, cKr=%d, cKc=%d, cNi=%d, cB=%d)\n",cRo,cCo, cNo, cKr, cKc, cNi, cB );
								int inIdx = cB + cNi*B + (cCo+cKc)*B*Ni + (cRo+cKr)*(Co+K-1)*B*Ni;
								//if(cRo==0&&cCo==18&&cB==0) printf("inIdx=%d, input=%lf\n", inIdx, *(input+inIdx));
								int outIdx = cB + cNo*B + cCo*B*No + cRo*Co*No*B;
								//if(cRo==0&&cCo==18&&cB==0) printf("outIdx=%d, output=%lf\n", outIdx, *(output+outIdx));
								int weightIdx = cNi + cNo*Ni + (cKr*K + cKc)*No*Ni;
								//if(cRo==0&&cCo==18&&cB==0) printf("weightIdx=%d, weight=%lf\n", weightIdx, *(weight+weightIdx));
								*(output + outIdx) += *(input + inIdx) * *(weight + weightIdx);
							//	*(output + outGetIdx(cB, cNo, cCo, cRo)) += 
							//	*(input + inGetIdx(cB, cNi, cCo+cKc, cRo+cKr)) *
							//	*(weight + weightGetIdx(cNi, cKc, cKr, cNo));
							}
						}
					}

			}//cNo

		}
	printf("end for ... \n");
}

int convfunc(void *ptr)
{
	int i;
	athread_init();
	ConvData *param = (ConvData *) ptr;
	for(i=0; i<STEPS; ++i){
		athread_spawn(convforward_v5, param);
		athread_join();
	}
	pthread_exit(NULL);
}

//for Ni=4x No=8x Batch=128x
void swConvForward(ConvData* param){
	//TODO:calc _CoStride
	

	int icg;
	ConvData* params[NUM_CG];
	for(icg = 0 ; icg < NUM_CG ; icg ++)
		params[icg] = malloc(sizeof(ConvData));

	ConvData* tparam;
	//int cgRo = (param->_Ro+(NUM_CG-1))/4;
	int cgRo = (param->_Ro)/4;
	int addCGRo=param->_Ro-cgRo*4;
	//printf("!!!test addCGRo=%d, cgRo=%d\n", addCGRo, cgRo);
	for(icg = 0 ; icg < NUM_CG ; icg ++)
	{
		tparam = params[icg];
		tparam->_Ni = param->_Ni;
		tparam->_No = param->_No;
		tparam->_K = param->_K;
		//if(cgRo*(icg+1)>param->_Ro) tparam->_Ro = param->_Ro-cgRo*icg;
		//else tparam->_Ro = cgRo;
		if(icg<addCGRo){
			tparam->_Ro = (cgRo+1);
			tparam->input = param->input + icg * (cgRo+1) * param->_Ci * param->_Ni * param->_B; 
			tparam->output = param->output + icg * (cgRo+1) * param->_Co * param->_No * param->_B;
		}
		else{
			tparam->_Ro=cgRo;
			tparam->input = param->input + (icg * cgRo + addCGRo) * param->_Ci * param->_Ni * param->_B; 
			tparam->output = param->output + (icg * cgRo + addCGRo) * param->_Co * param->_No * param->_B;
		}
		tparam->_Co = param->_Co;
		tparam->_Ri = tparam->_Ro + tparam->_K - 1; // Ro/4 + K-1
		tparam->_Ci = param->_Ci;
		tparam->_B = param->_B;
		tparam->_Costride = param->_Costride;
		tparam->weight = param->weight;
	}

	pthread_t ptr[NUM_CG];
	for(icg = 0 ; icg < NUM_CG ; icg ++)
	{
		pthread_create(&ptr[icg], NULL, (void*)convfunc, (void *)params[icg]);
	}

	for(icg = 0 ; icg < NUM_CG ; icg ++)
	{
		pthread_join(ptr[icg], NULL);
	}
}

//padding will change Ri&Ci
void swZeroPad(ConvData* param){
	int cB = param->_B;
	int cNi = param->_Ni;
	int cRo = param->_Ro;
	int cCo = param->_Co;
	int padn = param->_Pad;
	int pRi = param->_Ro+2*padn;
	int pCi = param->_Co+2*padn;
	if(pRi==param->_Ri&&pCi==param->_Ci) return;
	Type* pinput = (Type*)malloc(cB*cNi*pRi*pCi*sizeof(Type));
	Type* toutput = param->output;
	param->output = pinput;
	
	athread_spawn(zeroPad_v1, param);
	athread_join();

	free(param->input);
	param->input = pinput;
	param->output = toutput;
	param->_Ri = pRi;
	param->_Ci = pCi;
}

void zeroPad(ConvData* param){
	int B = param->_B;
	int Ni = param->_Ni;
	int Ri = param->_Ri;
	int Ci = param->_Ci;
	int padn = param->_Pad;
	int pRi = param->_Ri+2*padn;
	int pCi = param->_Ci+2*padn;
	Type* pinput = (Type*)malloc(B*Ni*pRi*pCi*sizeof(Type));
	int i,j;
	for(i=0;i<pRi;i++){
		for(j=0;j<pCi;j++){
			int ii=i-padn;
			int jj=j-padn;
			if(ii<0 || jj<0 || ii>=param->_Ri || jj>=param->_Ci){
				memset(pinput+i*pCi*Ni*B+j*Ni*B, 0, Ni*B*sizeof(Type));
			}else{
				memcpy(pinput+i*pCi*Ni*B+j*Ni*B, 
					   param->input+ii*param->_Ci*Ni*B+jj*Ni*B,
					   Ni*B*sizeof(Type));
			}
		}
	}
	free(param->input);
	param->input = pinput;
	param->_Ri = pRi;
	param->_Ci = pCi;
	return;
}

void convAddBias(ConvData* param, Type* output){
	Type* input = param->output;
	Type* bias = param->bias;
	int B = param->_B;
	int No = param->_No;
	int Ro = param->_Ro;
	int Co = param->_Co;

	int cB, cNo, cRo, cCo;
	for(cRo = 0; cRo<Ro; cRo++){
		for(cCo = 0; cCo<Co; cCo++){
			for(cNo = 0; cNo<No; cNo++){
				for(cB = 0; cB<B; cB++){
					output[cRo*Co*No*B+cCo*No*B+cNo*B+cB] = bias[cNo] + input[cRo*Co*No*B+cCo*No*B+cNo*B+cB];
				}
			}
		}
	}
}
void swConvAddBias(ConvData* param){
	
	athread_spawn(convAddBias_v1, param);
	athread_join();

}
