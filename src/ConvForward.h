/*************************************************************************
	> File Name: ./ConvForward.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Sat 03 Sep 2016 03:24:44 PM CST
 ************************************************************************/

#ifndef _CONVFORWARD_H_
#define _CONVFORWARD_H_

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include "def.h"

#define STEPS 10 
extern SLAVE_FUN(convforward_v5)();
extern SLAVE_FUN(zeroPad_v0)();
extern SLAVE_FUN(zeroPad_v1)();
extern SLAVE_FUN(convAddBias_v0)();
extern SLAVE_FUN(convAddBias_v1)();

void CaffeConv(ConvData* param);
void swConvForward(ConvData* param);

void swZeroPad(ConvData* param);
void zeroPad(ConvData* param);

void convAddBias(ConvData* param, Type* output);
void swConvAddBias(ConvData* param);

#endif
