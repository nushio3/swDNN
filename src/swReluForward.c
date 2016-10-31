/*************************************************************************
	> File Name: ./swReLUForward.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Sat 03 Sep 2016 10:32:15 AM CST
 ************************************************************************/

#include <unistd.h>
#include <slave.h>
#include "def.h"
#include <math.h>
#include <dma.h>

void reluForward_v1(ConvData* param){
	int B = param->_B;
	int Ni = param->_Ni;
	int No = param->_No;
	if(Ni!=No) return;
	//int Ro = param->_Ro;
	//int Co = param->_Co;
	
	int id = athread_get_id(-1);
	
	//load/store B 
	Type* buff = (Type*) ldm_malloc(sizeof(Type)*B);
	volatile int  replyget = 0, replyput = 0;

	int index = id;
	int cB;
	while(index<No){
		//load 1 B, store 1 B
		athread_get(PE_MODE, (void*)(param->input+index*B), (void*)buff, B*sizeof(Type),(void*) &replyget, 0,0,0);
		while(replyget!=1);
		replyget=0;
		for(cB=0; cB<B; cB++){
			if(buff[cB]<0){
				buff[cB] = 0;
			}
		}
		athread_put(PE_MODE,(void*)buff,(void*)(param->output+index*B),B*sizeof(Type), (void*)&replyput, 0,0);
		while(replyput!=1);
		replyput=0;
		index+=64;
	}

//	while(cRo<Ro){
//		while(cCo<Co){
//			//load 1*No*B, store 1*No*B
//			int index = cRo*Co*No*B+cCo*No*B;
//			int cNo, cB;
//			//load and store
//			for(cNo=0; cNo<No; cNo++){
//
//				athread_get(PE_MODE, (void*)(param->output+index+cNo*B), (void*)buff, B*sizeof(Type),(void*) &replyget, 0,0,0);
//				while(replyget!=1);
//				replyget=0;
//				for(cB=0; cB<B; cB++){
//					if(buff[cB]<0){
//						buff[cB] = 0;
//					}
//				}
//				athread_put(PE_MODE,(void*)buff,(void*)(param->output+index+cNo*B),B*sizeof(Type), (void*)&replyput, 0,0);
//				while(replyput!=1);
//				replyput=0;
//			}
//			cCo+=8;
//		}
//		cRo+=8;
//		cCo=cid;
//	}
}
void reluForward_v0(ConvData* param){
	int B = param->_B;
	int No = param->_No;
	int Ro = param->_Ro;
	int Co = param->_Co;
	
	int id = athread_get_id(-1);
	int cid = id%8, rid = id/8;
	int cRo=rid;
	int cCo=cid;
	
	//load/store B 
	Type* buff = (Type*) ldm_malloc(sizeof(Type)*B);
	volatile int  replyget = 0, replyput = 0;
	dma_desc dma_get_input, dma_put_output;
  
	dma_set_op(&dma_get_input, DMA_GET);
	dma_set_mode(&dma_get_input, PE_MODE);
	dma_set_reply(&dma_get_input, &replyget);

	dma_set_op(&dma_put_output, DMA_PUT);
	dma_set_mode(&dma_put_output, PE_MODE);
	dma_set_reply(&dma_put_output, &replyput);
  
	dma_set_size(&dma_get_input, B*sizeof(Type));
	//dma_set_bsize(&dma_get_input, B*sizeof(Type));
	//dma_set_stepsize(&dma_get_input, 0);
	
	dma_set_size(&dma_put_output, B*sizeof(Type));
	//dma_set_bsize(&dma_put_output, B/8*sizeof(Type));
	//dma_set_stepsize(&dma_put_output, B/8*7*sizeof(Type));
	
	while(cRo<Ro){
		while(cCo<Co){
			//load 1*No*B, store 1*No*B
			int index = cRo*Co*No*B+cCo*No*B;
			//int put_index = cRo*(Ci+2*Np)*Ni*B+cCo*Ni*B;
			int cNo, cB;
			//load and store
			for(cNo=0; cNo<No; cNo++){
				dma(dma_get_input, (long)(param->output+index+cNo*B), (long)buff);
				dma_wait(&replyget, 1); replyget=0;
				for(cB=0; cB<B; cB++){
					if(buff[cB]<0){
						buff[cB] = 0;
					}
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
