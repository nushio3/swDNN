/*************************************************************************
	> File Name: ./mlp.h
	> Author: zhaowl 
	> mail: cryinlaugh@126.com
	> Created Time: Tue 06 Sep 2016 10:42:11 AM CST
 ************************************************************************/

#ifndef _MLP_H_
#define _MLP_H_

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include "def.h"

void mlpForward(ConvData* param);
//void mlpForward(ConvData* param, Type* output);
void swMlpForward(ConvData* param);
void mlpBackward(ConvData* param, Type* tweight_delta, Type* tinput_diff);
void svMlpBackward(ConvData* param);
extern int dgemm_(char* transa, char* transb, int* m, int* n, int* k, Type* alpha, Type* a, int* lda, Type* b, int* ldb, Type* beta, Type* c, int* ldc);

#endif
