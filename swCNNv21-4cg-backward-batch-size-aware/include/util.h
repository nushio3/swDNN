/*************************************************************************
	> File Name: util.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 11:09:22 AM CST
 ************************************************************************/
#ifndef UTIL_H_
#define UTIL_H_

typedef double Type;

typedef struct ConvData_st{
  Type* input; //0
  Type* weight; //8
  Type* output; //16
  //   24,  28,  32,  36, 40,  44,  48, 52, 56 
  int _Ni, _Ri, _Ci, _No, _K, _Ro, _Co, _B, _Costride, _bCo;
}ConvData;


#endif
