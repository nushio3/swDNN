/*************************************************************************
	> File Name: ./def.c
	> Author: zhaowl 
	> mail: cryinlaugh@126.com
	> Created Time: Tue 27 Sep 2016 09:06:48 AM CST
 ************************************************************************/

#include "def.h"
#include <stdlib.h>
#include <stdio.h>
ConvData* newConvData(){
	ConvData* data = (ConvData*) malloc(sizeof(ConvData));
	data->input = NULL;
	data->weight = NULL;
	data->output = NULL;
	data->bias = NULL;
	data->output_diff = NULL;
	data->input_diff = NULL;
	data->weight_delta = NULL;
	data->bias_delta = NULL;
	data->label=NULL;
	Type loss = 0.0;
	int _Ni = 0;
	int _Ri = 0;
	int _Ci = 0;
	int _No = 0;
	int _K = 0;
	int _Ro = 0;
	int _Co = 0;
	int _B = 0;
	int _Costride = 0;
	int _Pad = 0;
	int _Biasize = 0;
	return data;
}
NetworkData* newNetworkData(){
	NetworkData* data = (NetworkData*) malloc(sizeof(NetworkData));
	data->_B = 0;
	data->nLayer = 0;
	data->totalMemSize = 0;
	data->LayerType = NULL;
	data->Config = NULL;
	data->Layers = NULL;
	data->InputData = NULL;
	data->LayerData = NULL;
	data->DiffData = NULL;
	data->Label = NULL;
	data->Weight = NULL;
	data->Bias = NULL;
	return data;
}

