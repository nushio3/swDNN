/*************************************************************************
	> File Name: ./netconfig.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 23 Sep 2016 09:22:13 AM CST
 ************************************************************************/

#ifndef _NETCONFIG_H_
#define _NETCONFIG_H_

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include "def.h"

void initNetwork(NetworkData* network);

const int nlayer = 8;
const int BatchSize = 128;
const char* layerType[] = {"conv", "relu", "conv","relu", "conv","relu", "mlp", "softmax"};
const int config[8][10] = {
	{48, 19, 19, 96, 19, 19, 5, 1, 2, 96},
	{96*19*19, 96*19*19, 0, 0, 0, 0, 0, 0, 0, 0},
	{96, 19, 19, 192, 19, 19, 3, 1, 1, 192},
	{192*19*19, 192*19*19, 0,0,0,0,0,0,0,0},
	{192, 19, 19, 1, 19, 19, 1, 1, 0, 361},
	{361, 361,0,0,0,0,0,0,0,0},
	{361, 361, 361,0,0,0,0,0,0,0},
	{361, 361,0,0,0,0,0,0,0,0}
};

#endif
