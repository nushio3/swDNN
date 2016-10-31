/*************************************************************************
	> File Name: ./netconfig.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 23 Sep 2016 09:27:15 AM CST
 ************************************************************************/

#include "netconfig.h"

void initLayerData(NetworkData* network);
void initWeightBiasData_Go(NetworkData* network);
void loadGoDataBatch(Type* data, int* label, int B, int Ni, int Ri, int Ci);
void initNetwork(NetworkData* network){
	//TODO:
	//char* config_fn = "network.conf";
	//FILE *pf = fopen(config_fn, "r+");
	//fclose(pf);
	
	//define network configurations
	
	//init NetworkData
	network->nLayer = nlayer;
	network->LayerType = (char**)layerType;
	printf("HERE test layer type\n");
	int it;
	network->Config = (int**)malloc(network->nLayer*sizeof(int*));
	for(it=0; it<network->nLayer; it++){
		network->Config[it] = (int*)&config[it][0];
	}
	network->_B = BatchSize;
	//alloc data in network 
	initLayerData(network);	
	//TODO:init weights and bias
	initWeightBiasData_Go(network);
}
void initLayerData(NetworkData* network){
	printf("netconfig.c: Enter initLayerData\n");
	int B = network->_B;
	int Ni, Ri, Ci, No, Ro, Co, K;
	int nl=0;
	int it, jt, cNi, cRi, cCi, cNo, cRo, cCo, cK;
	//load input data and label
	int inputDataSize, labelSize, tmpDataSize;
	Type* tmpData;
	if(network->LayerType[0] == "conv"){
		Ni = network->Config[0][0];
		Ri = network->Config[0][1];
		Ci = network->Config[0][2]; 
		No = network->Config[network->nLayer-1][1];
		inputDataSize = B*Ni*Ri*Ci;
		labelSize = B*No;
		network->InputData = (Type*) malloc(sizeof(Type)*inputDataSize);
		network->totalMemSize += inputDataSize*sizeof(Type);
		network->Label = (int*) malloc(sizeof(int)*labelSize);
		network->totalMemSize += labelSize*sizeof(int);
		loadGoDataBatch(network->InputData, network->Label, B, Ni, Ri, Ci);
	}else{
		printf("Unsupported layer type for label!\n");
		exit(1);
	}

	printf("netconfig.c: Finish input data and label init\n");

	//do mem alloc for each layers
	network->LayerData = (Type**) malloc(sizeof(Type*)*network->nLayer);
	network->DiffData = (Type**) malloc(sizeof(Type*)*network->nLayer);
	network->Weight = (Type**) malloc(sizeof(Type*)*network->nLayer);
	network->WeightDelta = (Type**) malloc(sizeof(Type*)*network->nLayer);
	network->Bias = (Type**) malloc(sizeof(Type*)*network->nLayer);
	network->BiasDelta = (Type**) malloc(sizeof(Type*)*network->nLayer);
	for(nl=0; nl<network->nLayer; nl++){
		printf("netconfig.c: alloc layer %d data\n", nl);
		switch(network->LayerType[nl][0]){
			case 'c': 
				Ni = network->Config[nl][0];
				No = network->Config[nl][3];
				Ro = network->Config[nl][4];
				Co = network->Config[nl][5];
				K  = network->Config[nl][6];
				
				//alloc weight
				tmpDataSize = Ni*No*K*K;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->Weight[nl] = tmpData;
				
				//alloc weight delta
				tmpDataSize = Ni*No*K*K;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->WeightDelta[nl] = tmpData;
				
				//alloc bias
				tmpDataSize = network->Config[nl][9];
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->Bias[nl] = tmpData;

				//alloc bias delta
				tmpDataSize = network->Config[nl][9];
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->BiasDelta[nl] = tmpData;
				
				//alloc output data
				tmpDataSize = B*No*Ro*Co;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->LayerData[nl] = tmpData;

				//alloc diff data
				tmpDataSize = B*No*Ro*Co;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->DiffData[nl] = tmpData;
				break;
			case 'r': 
				Ni = network->Config[nl][0];
				No = network->Config[nl][1];
				
				//no weights and bias
				network->Weight[nl] = NULL;
				network->Bias[nl] = NULL;

				//alloc output data
				tmpDataSize = B*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->LayerData[nl] = tmpData;
				
				//alloc diff data
				tmpDataSize = B*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->DiffData[nl] = tmpData;
				break;
			case 'm': 
				Ni = network->Config[nl][0];
				No = network->Config[nl][1];
				
				//alloc weights
				tmpDataSize = Ni*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->Weight[nl] = tmpData;
				
				//alloc weights delta
				tmpDataSize = Ni*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->WeightDelta[nl] = tmpData;
				
				//alloc bias
				tmpDataSize = network->Config[nl][2];
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->Bias[nl] = tmpData;
				
				//alloc bias delta
				tmpDataSize = network->Config[nl][2];
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->BiasDelta[nl] = tmpData;

				//alloc output data
				tmpDataSize = B*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->LayerData[nl] = tmpData;
				
				//alloc diff data
				tmpDataSize = B*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->DiffData[nl] = tmpData;
				break;
			case 's': 
				Ni = network->Config[nl][0];
				No = network->Config[nl][1];
				
				//no weights and bias
				network->Weight[nl] = NULL;
				network->Bias[nl] = NULL;
				
				//alloc output data
				tmpDataSize = B*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->LayerData[nl] = tmpData;

				//alloc diff data
				tmpDataSize = B*No;
				tmpData = (Type*) malloc(sizeof(Type)*tmpDataSize);
				network->totalMemSize += tmpDataSize*sizeof(Type);
				network->DiffData[nl] = tmpData;
				break;
			default: 
				printf("Unsupported layer type for data init!\n");
				exit(1);

		}
	}
	printf("netconfig.c: Finish alloc layer data. Total memory usage: %lf GB\n", network->totalMemSize/1024.0/1024.0/1024.0);
	
}
void initWeightBiasData_Go(NetworkData* network){
	int B = network->_B;
	int Ni, Ri, Ci, No, Ro, Co, K;
	int nl=0;
	int it, jt, cNi, cRi, cCi, cNo, cRo, cCo, cK;
	//load input data and label
	int tmpDataSize;
	Type* tmpData;
	
	//init weight&bias for each layer
	for(nl=0; nl<network->nLayer; nl++){
		switch(network->LayerType[nl][0]){
			case 'c': 
				Ni = network->Config[nl][0];
				No = network->Config[nl][3];
				Ro = network->Config[nl][4];
				Co = network->Config[nl][5];
				K  = network->Config[nl][6];
				
				//init weight
				tmpDataSize = Ni*No*K*K;
				tmpData = network->Weight[nl];
				if(nl==0||nl==2){
					for(it=0; it<tmpDataSize; it++){
						tmpData[it] = 1e-4;
					}
				}else if(nl==4){
					for(it=0; it<tmpDataSize; it++){
						tmpData[it] = 0.04;
					}
				}else{
					printf("Unsupported weight initialization!\n");
					exit(1);
				}
				
				//init bias
				tmpDataSize = network->Config[nl][9];
				tmpData = network->Bias[nl];
				if(nl==0){
					for(it=0; it<tmpDataSize; it++){
						tmpData[it] = 0.0;
					}
				}else if(nl==2||nl==4){
					for(it=0; it<tmpDataSize; it++){
						tmpData[it] = 0.1;
					}
				}else{
					printf("Unsupported weight initialization!\n");
					exit(1);
				}
				break;
			case 'r': 
				break;
			case 'm': 
				Ni = network->Config[nl][0];
				No = network->Config[nl][1];
				
				//alloc weights
				tmpDataSize = Ni*No;
				tmpData = network->Weight[nl];
				for(it=0; it<tmpDataSize; it++){
					tmpData[it] = 1.0/361;
				}

				//alloc bias
				tmpDataSize = network->Config[nl][2];
				tmpData = network->Bias[nl];
				for(it = 0; it<tmpDataSize; it++){
					tmpData[it] = 0;
				}
				break;
			case 's': 
				break;
			default: 
				printf("Unsupported layer type for data init!\n");
				exit(1);

		}
	}	
}
void loadGoDataBatch(Type* data, int* label, int B, int Ni, int Ri, int Ci){
	printf("Enter loadGoDataBatch\n");
	if(Ri*Ci!=361){
		printf("Wrong input feature map size. Expected:%dx%d\n", 19, 19);
		exit(1);
	}
	char* fn_data = "./data/13.bin";
	FILE* fp = fopen(fn_data, "r+");
	int it,jt;
	char tlabel[3];
	int ilabel;
	char tmap[361];
	int cB, cNi;
	for(cB=0; cB<B; cB++){
		//read label
		fread(tlabel, 3, 1, fp);
		ilabel=(tlabel[0]-48)*100+(tlabel[1]-48)*10+(tlabel[2]-48);
		for(it=0; it<Ri*Ci; it++){
			if(it==ilabel) label[it*B+cB] = 1;
			else label[it*B+cB] = 0;
		}
		//printf("batchn = %d, position = %ld, lable = %d\n", cB, ftell(fp), label[cB]);
		//read data
		for(cNi=0; cNi<Ni; cNi++){
			fread(tmap,Ri*Ci, 1, fp);
			for(it=0; it<Ri*Ci; it++){
				*(data+cNi*Ri*Ci*B+it*B+cB) = (Type) tmap[it];
			}
		}
	}
	fclose(fp);
	printf("Finish load Go batch data.\n");
}
