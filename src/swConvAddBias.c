/*************************************************************************
	> File Name: ./swConvAddBias.c
	> Author: zhaowl 
	> mail: cryinlaugh@126.com 
	> Created Time: Tue 06 Sep 2016 09:03:09 AM CST
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "def.h"
#include <dma.h>

void convAddBias_v1(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	int Ro = param->_Ro;
	int Co = param->_Co;

	int id = athread_get_id(-1);
	int cid = id%8, rid = id/8;
	int cRo=rid;
	int cCo=cid;
	//printf("here is %d", id);
	
	//load/store B 
	Type* buff = (Type*)ldm_malloc(sizeof(Type)*B);
	Type* bias = (Type*)ldm_malloc(sizeof(Type)*No);

	volatile int replyget = 0, replyput = 0;

	athread_get(PE_MODE, (void*)(param->bias), (void*)bias, No*sizeof(Type),(void*) &replyget, 0,0,0);
	while(replyget!=1);
	replyget=0;


	int cNo,cB, index;
	while(cRo<Ro){
		while(cCo<Co){
			//load 1*Ni*B, store 1*Ni*B
			//int get_index = (cRo-Np)*Ci*Ni*B+(cCo-Np)*Ni*B;
			index = cRo*(Co)*No*B+cCo*No*B;
			//load and store
			for(cNo=0; cNo<No; cNo++){
				athread_get(PE_MODE, (void*)(param->output+index+cNo*B), (void*)buff, B*sizeof(Type),(void*) &replyget, 0,0,0);
				while(replyget!=1);
				replyget=0;
				for(cB=0; cB<B; cB++){
					buff[cB]+=bias[cNo];
				}
				athread_put(PE_MODE,(void*)buff,(void*)(param->output+index+cNo*B),B*sizeof(Type), (void*)&replyput, 0,0);
				while(replyput!=1);
				replyput=0;
			}
			cCo+=8;
		}
		cRo+=8;
		cCo=cid;
	}
}
void convAddBias_v0(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	int Ro = param->_Ro;
	int Co = param->_Co;

	int id = athread_get_id(-1);
	int cid = id%8, rid = id/8;
	int cRo=rid;
	int cCo=cid;
	//printf("here is %d", id);
	
	//load/store B 
	Type* buff = (Type*)ldm_malloc(sizeof(Type)*B);
	Type* bias = (Type*)ldm_malloc(sizeof(Type)*No);

	volatile int  replyget = 0, replyput = 0;
	dma_desc dma_get_input, dma_put_output;
  
	dma_set_op(&dma_get_input, DMA_GET);
	dma_set_mode(&dma_get_input, PE_MODE);
	dma_set_reply(&dma_get_input, &replyget);

	dma_set_op(&dma_put_output, DMA_PUT);
	dma_set_mode(&dma_put_output, PE_MODE);
	dma_set_reply(&dma_put_output, &replyput);

	dma_set_size(&dma_get_input, No*sizeof(Type));
	
	dma(dma_get_input, (long)(param->bias), (long)bias);
	dma_wait(&replyget, 1); replyget=0;

	dma_set_size(&dma_get_input, B*sizeof(Type));
	
	dma_set_size(&dma_put_output, B*sizeof(Type));

	int cNo,cB, index;
	while(cRo<Ro){
		while(cCo<Co){
			//load 1*Ni*B, store 1*Ni*B
			//int get_index = (cRo-Np)*Ci*Ni*B+(cCo-Np)*Ni*B;
			index = cRo*(Co)*No*B+cCo*No*B;
			//load and store
			for(cNo=0; cNo<No; cNo++){
				dma(dma_get_input, (long)(param->output+index+cNo*B), (long)buff);
				dma_wait(&replyget, 1); replyget=0;
				for(cB=0; cB<B; cB++){
					buff[cB]+=bias[cNo];
				}
				dma(dma_put_output, (long)(param->output+index+cNo*B), (long)buff);
				dma_wait(&replyput, 1); replyput=0;
			}
			cCo+=8;
		}
		cRo+=8;
		cCo=cid;
	}
}
