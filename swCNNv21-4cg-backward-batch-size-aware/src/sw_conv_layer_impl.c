/*************************************************************************
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 04:12:22 PM CST
 ************************************************************************/

#include<stdio.h>
#include<assert.h>
#include "athread.h"
#include "util.h"
#include "sw_conv_layer_impl.h"

extern SLAVE_FUN(conv_valid)();
extern SLAVE_FUN(conv_full)();

void sw_conv_forward_impl(Type* in, Type* weight, Type* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{/*{{{*/

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    param->input = in;
    param->weight = weight;
    param->output = out;
	param->_Ni = Ni;
	param->_Ri = Ri;
	param->_Ci = Ci;
	param->_No = No;
	param->_K = K;
	param->_Ro = Ri-K+1;
	param->_Co = Ci-K+1;
	param->_B = B;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

	int Costride = (64*55*1024/8-Ni*B*2-Ni*No*K*2)/(No*B)-(K-1);
	param->_Costride = Costride;
    assert(Costride > 0);
	int ldm_consume = 8*(Ni*No*K*2 + No*B*(Costride+K-1) + Ni*B*2);
	printf("ldm comsumption is %d\n", ldm_consume/64);
	assert(ldm_consume < 64*1024*64);

    athread_init();
	athread_spawn(conv_valid, param);
	athread_join();

    free(param);
}/*}}}*/

void sw_conv_backward_impl(Type* in, 
        Type* in_grad,
        Type* out_grad,
        Type* weight_diff,
        Type* weight,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    Type* in_T = (Type*)malloc(sizeof(Type)*Ri*Ci*Ni*B);

    //Transformation: in (B, Ni) -> (Ni, B)
    //TODO: Can be acc with CPEs
    int cRi, cCi, cNi, cB;
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                    *(in_T+cB*Ni+cNi+cCi*B*Ni+cRi*Ci*Ni*B) = 
                        *(in+cB+cNi*B+cCi*B*Ni+cRi*Ci*Ni*B);

    param->input = in_T;
    param->weight = out_grad;
    param->output = weight_diff;
	param->_Ni = B;
	param->_Ri = Ri;
	param->_Ci = Ci;
	param->_No = No;
	param->_K  = Ci-K+1;
	param->_Ro = K;
	param->_Co = K;
	param->_B  = Ni;
    

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

	int Costride = (64*55*1024/8-param->_Ni*param->_B*2-
            param->_Ni*param->_No*param->_K)/
        (param->_No*param->_B)-(param->_K-1);
	param->_Costride = Costride;
    assert(Costride > 0);
	int ldm_consume = 8*(Ni*No*K*2 + No*B*(Costride+K-1) + Ni*B*2);
	printf("ldm comsumption is %d\n", ldm_consume/64);
	assert(ldm_consume < 64*1024*64);

    athread_init();
	athread_spawn(conv_valid, param);
	athread_join();

    //in_grad
    //Transforamation Weight
    Type* weight_T   = (Type*)malloc(sizeof(Type)*No*Ni*K*K);
    int cKr, cKc, cNo;
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                    *(weight_T + cNi*No + cNo + cKc*Ni*No + cKr*Ni*No*K) =
                        *(weight + cNi + cNo*Ni + cKc*Ni*No + cKr*Ni*No*K);
                }
        
    param->input =  out_grad;
    param->weight = weight_T;
    param->output = in_grad;
	param->_Ni = No;
	param->_Ri = Ri-K+1;
	param->_Ci = Ci-K+1;
	param->_No = Ni;
	param->_K  = K;
	param->_Ro = Ri;
	param->_Co = Ci;
	param->_B  = B;

    Costride = (64*55*1024/8-param->_Ni*param->_B*2-param->_Ni*param->_No*2)/
            (param->_No*param->_B)-(param->_K-1);
	param->_Costride = Costride;
	printf("Costride is %d\n", Costride);
    assert(Costride > 0);

	athread_spawn(conv_full, param);
	athread_join();


    free(param);
    free(in_T);
    free(weight_T);
}
