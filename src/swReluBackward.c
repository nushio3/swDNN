/*************************************************************************
	> File Name: ./swReluBackward.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Wed 21 Sep 2016 09:56:21 AM CST
 ************************************************************************/

#include <unistd.h>
#include <slave.h>
#include "def.h"
#include <math.h>
#include <dma.h>

void reluBackward_v1(ConvData* param){
	int B = param->_B;
	int Ni = param->_Ni;
	int No = param->_No;
	if(Ni!=No) return;
	//int Ro = param->_Ro;
	//int Co = param->_Co;
	
	int id = athread_get_id(-1);
	
	//load/store 1*B input/output_diff 
	Type* buff = (Type*) ldm_malloc(sizeof(Type)*B);
	Type* buff_diff = (Type*) ldm_malloc(sizeof(Type)*B);
	volatile int  replyget = 0, replyput = 0;
	
	int index = id;
	int cB;
	while(index<No){
		//load 1*No*B, store 1*No*B
		athread_get(PE_MODE, (void*)(param->input+index*B), (void*)buff, B*sizeof(Type),(void*) &replyget, 0,0,0);
		while(replyget!=1);
		replyget=0;
		athread_get(PE_MODE, (void*)(param->output_diff+index*B), (void*)buff_diff, B*sizeof(Type),(void*) &replyget, 0,0,0);
		while(replyget!=1);
		replyget=0;
		for(cB=0; cB<B; cB++){
			if(buff[cB]<= 0){
				buff_diff[cB] = 0;
			}
		}
		athread_put(PE_MODE,(void*)buff_diff,(void*)(param->input_diff+index*B),B*sizeof(Type), (void*)&replyput, 0,0);
		while(replyput!=1);
		replyput=0;
		index+=64;	
	}
}
//void reluBackward_v1(ConvData* param){
//	int B = param->_B;
//	int No = param->_No;
//	int Ro = param->_Ro;
//	int Co = param->_Co;
//	
//	int id = athread_get_id(-1);
//	int cid = id%8, rid = id/8;
//	int cRo=rid;
//	int cCo=cid;
//
//	
//	//load/store B 
//	Type* buff = (Type*) ldm_malloc(sizeof(Type)*B);
//	Type* buff_diff = (Type*) ldm_malloc(sizeof(Type)*B);
//	volatile int  replyget = 0, replyput = 0;
//	
//	while(cRo<Ro){
//		while(cCo<Co){
//			//load 1*No*B, store 1*No*B
//			int index = cRo*Co*No*B+cCo*No*B;
//			int cNo, cB;
//			for(cNo=0; cNo<No; cNo++){
//				athread_get(PE_MODE, (void*)(param->output+index+cNo*B), (void*)buff, B*sizeof(Type),(void*) &replyget, 0,0,0);
//				while(replyget!=1);
//				replyget=0;
//				athread_get(PE_MODE, (void*)(param->output_diff+index+cNo*B), (void*)buff_diff, B*sizeof(Type),(void*) &replyget, 0,0,0);
//				while(replyget!=1);
//				replyget=0;
//				for(cB=0; cB<B; cB++){
//					if(buff[cB]<= 0){
//						buff_diff[cB] = 0;
//					}
//				}
//				athread_put(PE_MODE,(void*)buff_diff,(void*)(param->output_diff+index+cNo*B),B*sizeof(Type), (void*)&replyput, 0,0);
//				while(replyput!=1);
//				replyput=0;
//			}
//			cCo+=8;
//		}
//		cRo+=8;
//		cCo=cid;
//	}
//}
