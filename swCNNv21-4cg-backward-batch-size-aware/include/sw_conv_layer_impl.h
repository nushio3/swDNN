/*************************************************************************
	> File Name: sw_conv_forward_impl.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 04:17:22 PM CST
 ************************************************************************/
#ifndef SW_CONV_FORWARD_IMPL_H_
#define SW_CONV_FORWARD_IMPL_H_

#include "util.h"
void sw_conv_forward_impl(Type* in, Type* weight, Type* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B);

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
        int B);
#endif
