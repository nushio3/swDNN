/*************************************************************************
	> File Name: ./swZeroPad.c
	> Author: zhaowl 
	> mail: cryinlaugh@126.com
	> Created Time: Thu 01 Sep 2016 03:12:55 PM CST
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "def.h"
#include <dma.h>

void zeroPad_v1(ConvData* param){
	int B = param->_B;
	int Ni = param->_Ni;
	int Ri = param->_Ri;
	int Ci = param->_Ci;
	int Np = param->_Pad;

	int id = athread_get_id(-1);
	int cid = id%8, rid = id/8;
	int cRo=rid;
	int cCo=cid;
	//printf("here is %d", id);
	
	//load/store B 
	Type* buff = (Type*) ldm_malloc(sizeof(Type)*B);

	volatile int  replyget = 0, replyput = 0;
	
	while(cRo<Ri+2*Np){
		while(cCo<Ci+2*Np){
			//load 1*Ni*B, store 1*Ni*B
			int get_index = (cRo-Np)*Ci*Ni*B+(cCo-Np)*Ni*B;
			int put_index = cRo*(Ci+2*Np)*Ni*B+cCo*Ni*B;
			int i;
			if(cRo<Np || cCo<Np || cRo >= Ri+Np || cCo>=Ci+Np){
				//store 0
				//memset(buff,0, B*sizeof(Type));
				for(i=0; i<B; i++){
					buff[i]=(Type)0.0;
				}
				for(i=0; i<Ni; i++){
					athread_put(PE_MODE,(void*)buff,(void*)(param->output+put_index+i*B),B*sizeof(Type), (void*)&replyput, 0,0);
					while(replyput!=1);
					replyput=0;
					//dma(dma_put_output, (long)(param->output+put_index+i*B), (long)buff);
					//dma_wait(&replyput, 1); replyput=0;
				}
			}else{
				//load and store
				for(i=0; i<Ni; i++){
					athread_get(PE_MODE, (void*)(param->input+get_index+i*B), (void*)buff, B*sizeof(Type),(void*)&replyget, 0,0,0);
					while(replyget!=1);
					replyget=0;
					athread_put(PE_MODE,(void*)buff,(void*)(param->output+put_index+i*B),B*sizeof(Type), (void*)&replyput, 0,0);
					while(replyput!=1);
					replyput=0;
					//dma(dma_get_input, (long)(param->input+get_index+i*B), (long)buff);
					//dma_wait(&replyget, 1); replyget=0;
					//dma(dma_put_output, (long)(param->output+put_index+i*B), (long)buff);
					//dma_wait(&replyput, 1); replyput=0;
				}
			}
			cCo+=8;
		}
		cRo+=8;
		cCo=cid;
	}
}
