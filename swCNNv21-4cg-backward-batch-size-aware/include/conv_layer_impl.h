/*************************************************************************
	> File Name: ConvLayer_impl.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 10:24:37 AM CST
 ************************************************************************/
#ifndef CONVLAYER_IMPL
#define CONVLAYER_IMPL
#include "util.h"


void conv_forward_impl(Type* in, Type* weight, Type* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B);

void conv_backward_impl(Type* in,
        Type* out_grad, Type* weight,
        Type* in_grad, Type* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B);


#endif
