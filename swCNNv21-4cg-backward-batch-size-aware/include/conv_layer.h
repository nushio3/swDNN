/*************************************************************************
	> File Name: Layer.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 10:19:29 AM CST
 ************************************************************************/
#pragma once
#include<iostream>
#include"tensor.h"
#include"util.h"
#define SW_64
extern "C"{
#include"conv_layer_impl.h"
#ifdef SW_64
#include"sw_conv_layer_impl.h"
#endif
}
using namespace std;


class conv_layer{
public:
    tensor in;
    tensor in_grad;
    tensor weight;
    tensor weight_diff;

    int Ni, No, B, Ci, Ri, K;
    //init layer
    conv_layer(int _B, int _Ni, int _No, int _K, int _Ri, int _Ci):
        in(_B, _Ni, _Ci, _Ri),
        in_grad(_B, _Ni, _Ci, _Ri),
        weight(_Ni, _No, _K, _K),
        weight_diff(_Ni, _No, _K, _K),
        Ni(_Ni),No(_No),B(_B),Ci(_Ci),Ri(_Ri),K(_K)
    {
    }
  
    void init_data(tensor& T){
      T.rand_init();
    }

#ifdef SW_64
    void swforward(tensor& out){
        sw_conv_forward_impl(in.data, weight.data, out.data,
                Ci, Ri, K, Ni, No, B);
        cout<<"[64 CPE]Convolutional Layer Forward is OK!"<<endl;
    }

    void swbackward(tensor& out_grad){
        sw_conv_backward_impl(in.data, 
                in_grad.data,
                out_grad.data,
                weight_diff.data,
                weight.data,
            Ci, Ri, K, Ni, No, B);
        cout<<"[64 CPE]Convolutional Layer Backward is OK!"<<endl;
    }
#endif

    void forward(tensor& out){
        conv_forward_impl(in.data, weight.data, out.data,
                Ci, Ri, K, Ni, No, B);
        cout<<"Convolutional Layer Forward is OK!"<<endl;
    }

    void backward(tensor& out_grad){
        
        conv_backward_impl(in.data,
                           out_grad.data, 
                           weight.data,
                           in_grad.data,
                           weight_diff.data,
                Ci, Ri, K, Ni, No, B);
                

        cout<<"Convolutional Layer Backward is OK!"<<endl;
    }
};
