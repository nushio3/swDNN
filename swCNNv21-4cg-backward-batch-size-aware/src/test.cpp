/*************************************************************************
	> File Name: test.cpp
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 10:27:13 AM CST
 ************************************************************************/

#include<iostream>
using namespace std;

#include "conv_layer.h"
#include "util.h"

void forward_test();
void backward_test();

int main(){
    //forward_test();
    backward_test();
    return 0;
}

void forward_test()
{

    int B = 128;
    int Ni[2] = {128, 256};
    int No[2] = {256, 1};
    int K[2]  = {3, 3};
    int Ri[2] = {8, 6};
    int Ci[2] = {8, 6};

    conv_layer conv1(B, Ni[0], No[0], K[0], Ri[0], Ci[0]); 
    conv_layer conv2(B, Ni[1], No[1], K[1], Ri[1], Ci[1]); 
    cout<<"initialization is ok!"<<endl;

    conv1.in.rand_init();
    conv1.weight.rand_init();

    /* forwad test*/
    conv1.swforward(conv2.in);

    tensor cpu_out(B,Ni[1],Ci[1],Ri[1]);
    conv1.forward(cpu_out);

    if( cpu_out.compare(conv2.in) == true)
        cout<<"conv forward correct!"<<endl;
    else
        cout<<"conv forward wrong!"<<endl;
    conv2.in.store_tensor("./log/fp_sw.txt");    
    cpu_out.store_tensor("./log/fp_cpu.txt");    
}

void backward_test(){
    int B = 64;
    int Ni[2] = {128, 64};
    int No[2] = {64, 128};
    int K[2]  = {3, 3};
    int Ri[2] = {5, 3};
    int Ci[2] = {5, 3};

    conv_layer conv1(B, Ni[0], No[0], K[0], Ri[0], Ci[0]); 
    conv_layer conv2(B, Ni[1], No[1], K[1], Ri[1], Ci[1]); 
    conv2.in_grad.rand_init();
    conv1.in.rand_init();

    conv1.backward(conv2.in_grad);
    tensor cpu_weight_diff(conv1.weight_diff);
    tensor cpu_in_grad(conv1.in_grad);
    
    cout<<"CPU backward Over!"<<endl;

    conv1.swbackward(conv2.in_grad);

    conv1.weight_diff.store_tensor("./log/bp_weight_diff_sw.txt");    
    cpu_weight_diff.store_tensor("./log/bp_weight_diff_cpu.txt");    

    if( cpu_weight_diff.compare(conv1.weight_diff) == true)
        cout<<"conv backward weight_diff correct!"<<endl;
    else
        cout<<"conv backward weight_diff wrong!"<<endl;

    if( cpu_in_grad.compare(conv1.in_grad) == true)
        cout<<"conv backward in_grad correct!"<<endl;
    else
        cout<<"conv backward in_grad wrong!"<<endl;

}
