/*************************************************************************
	> File Name: ./ReLU.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Sat 03 Sep 2016 10:12:30 AM CST
 ************************************************************************/

#ifndef _RELU_H_
#define _RELU_H_
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include "def.h"

void reluForward(ConvData* param, Type* output);
void reluBackward(ConvData* param, Type* output);
void swReluForward(ConvData* param);
void swReluBackward(ConvData* param);
extern SLAVE_FUN(reluForward_v0)();
extern SLAVE_FUN(reluForward_v1)();
extern SLAVE_FUN(reluBackward_v0)();
extern SLAVE_FUN(reluBackward_v1)();
#endif
