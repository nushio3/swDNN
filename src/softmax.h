/*************************************************************************
	> File Name: ./softmax.h
	> Author: zhaowl 
	> mail: cryinlaugh@126.com
	> Created Time: Wed 07 Sep 2016 02:47:33 PM CST
 ************************************************************************/

#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "def.h"
//TODO:
//extern SLAVE_FUN(softmaxForward_v0)();
//void wsSoftmaxForward(SoftmaxData* param);

void softmaxForward(ConvData* param);
void softmaxWithLoss(ConvData* param);
void softmaxBackward(ConvData* param);

#endif
