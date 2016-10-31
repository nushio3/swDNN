#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "def.h"
#include <dma.h>
//#include <simd_test.h>

/***************
 * GEMM PLAN 
 * Jerry Fang 2016.6.16
 * 苟利国家生以死
 * ************/
//#define DEBUG
#define SIMDSIZE 4
void convforward_p_simd_rc_c(ConvData* param)
{
  int cB, cNi, cRi, cCi, cKr, cKc, ccCore, crCore, cNo;
  int ii, jj, cRo, cCo;
  int CoStart;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int input_calc_index=1, input_load_index=0;
  int i, j;
  int Ni, Ri, Ci, No, K, Ro, Co, B;
  Ni = param->_Ni;
  Ri = param->_Ri;
  Ci = param->_Ci;
  No = param->_No;
  K = param->_K;
  Ro = param->_Ro;
  Co = param->_Co;
  B = param->_B;
  int CStride=param->_Costride;

//B, Ni, Ci, Ri
  SIMDType* local_input  = (SIMDType*) ldm_malloc(sizeof(Type)*Ni*B/8/8*2);
  int local_input_size = Ni*B/8/8/SIMDSIZE;
//No, Ni, K, K
  Type* local_weight = (Type*) ldm_malloc(sizeof(Type)*Ni*No/8/8*K);
//B, No, Co, Ro
  SIMDType* local_output = (SIMDType*) ldm_malloc(sizeof(Type)*No*B/8/8*CStride);

//  Type local_weight[K*K*Ni/64*No];
//initilize DMA variables
  volatile int  replyget = 0, replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &replyget);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &replyget);

  dma_set_op(&dma_get_output, DMA_GET);
  dma_set_mode(&dma_get_output, PE_MODE);
  dma_set_reply(&dma_get_output, &replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(B/8, Ni/8)
  dma_set_size(&dma_get_input, B*Ni/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_get_input, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_get_input, B/SIMDSIZE/8*7*sizeof(SIMDType));

  //DMA for local_weight(Ni/8, No/8)
  dma_set_size(&dma_get_weight, No*Ni/8/8*sizeof(Type));
  dma_set_bsize(&dma_get_weight, Ni/8*sizeof(Type));
  dma_set_stepsize(&dma_get_weight, Ni/8*7*sizeof(Type));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_get_output, B*No/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_get_output, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_get_output, B/SIMDSIZE/8*7*sizeof(SIMDType));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_put_output, B*No/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_put_output, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_put_output, B/SIMDSIZE/8*7*sizeof(SIMDType));

  for(CoStart=0; CoStart<Co; CoStart+=CStride){
    int CoEnd = CoStart+CStride;
    int CiEnd = CoStart+CStride+K;
    if(CoEnd > Co)
      CoEnd = Co;
    if(CiEnd > Ci)
      CiEnd = Ci;
    //input init
    Type* input_ptr = param->input+cid*B/8+rid*Ni/8*B+(CoStart)*Ni*B;
    for(cRo=0; cRo<Ro; ++cRo){
      //DMA get output(:,:,:,cRo)
      jj=0;
      Type* output_ptr = param->output + cid*B/8 + rid*No/8*B + B*No*(cRo*Co+CoStart);
      for(ii=CoStart; ii<CoEnd; ++ii){
          dma(dma_get_output, (long)(output_ptr), (long)(local_output+jj*B*No/64/SIMDSIZE));
          dma_wait(&replyget, 1); replyget = 0;
	  output_ptr+=B*No;
          jj++;
      }
      output_ptr -= jj*B*No;

      Type* weight_ptr = param->weight+(cid*Ni/8+rid*No/8*Ni);
      for(cKr=0; cKr<K; ++cKr){
        cRi = cRo+cKr;
        //DMA get filter(:,:,:,cKr)
        for(ii=0; ii<K; ++ii){
          dma(dma_get_weight, (long)(weight_ptr), (long)(local_weight+ii*Ni*No/64));
          dma_wait(&replyget, 1); replyget = 0;
	  weight_ptr += Ni*No;
        }

#define CORE
	  //DMA 1st input
          dma(dma_get_input, (long)(input_ptr), (long)(local_input+input_calc_index*local_input_size));
          dma_wait(&replyget, 1); replyget = 0;
	  input_ptr += Ni*B;

          for(cCi=CoStart; cCi<CiEnd; ++cCi){
            //DMA for next line of input (B/8, Ni/8) 4*4
            dma(dma_get_input, (long)(input_ptr), (long)(local_input+input_load_index*local_input_size));
            
            for(cKc=0; cKc<K; ++cKc){
              cCo = cCi-cKc;

              if(cCo >= CoStart && cCo < CoEnd){

                SIMDType tmp_weight[4]; //= {0.0, 0.0, 0.0, 0.0};
                SIMDType tmp_input[4]; // = {0.0, 0.0, 0.0, 0.0};
	#ifdef CORE	
                for(ccCore=0; ccCore<8; ++ccCore){

		  SIMDType* local_output_ptr = local_output + (cCo-CoStart)*No*B/64/SIMDSIZE;
		  SIMDType* local_input_ptr  = local_input  + input_calc_index*local_input_size;
		  Type*	    local_weight_ptr = local_weight + cKc*Ni*No/64;

                  for(cNo= 0; cNo < No/8; cNo+=4){
		    for(cB = 0; cB < B/SIMDSIZE/8; cB+=4){
		      SIMDType tmp_output[16];
		      for(i =0; i<16; ++i)
			tmp_output[i] = 0.0;

		      for(cNi=0; cNi<Ni/8; ++cNi){
                        if(ccCore == rid){
                          tmp_input[0] = *(local_input_ptr++) ;
                          simd_putc(tmp_input[0],8);
                          tmp_input[1] = *(local_input_ptr++) ;
                          simd_putc(tmp_input[1],8);
                          tmp_input[2] = *(local_input_ptr++);
                          simd_putc(tmp_input[2],8);
                          tmp_input[3] = *(local_input_ptr);
                          simd_putc(tmp_input[3],8);
			  local_input_ptr -= 3;
                        }
                        else{
                          tmp_input[0] = simd_getc(tmp_input[0]);
                          tmp_input[1] = simd_getc(tmp_input[1]);
                          tmp_input[2] = simd_getc(tmp_input[2]);
                          tmp_input[3] = simd_getc(tmp_input[3]);
                        }
                        
                        if(ccCore == cid){
                          simd_loade(tmp_weight[0], local_weight_ptr);
                          simd_putr(tmp_weight[0], 8);
                          simd_loade(tmp_weight[1], local_weight_ptr+Ni/8);
                          simd_putr(tmp_weight[1], 8);
                          simd_loade(tmp_weight[2], local_weight_ptr+Ni/8*2);
                          simd_putr(tmp_weight[2], 8);
                          simd_loade(tmp_weight[3], local_weight_ptr+Ni/8*3);
                          simd_putr(tmp_weight[3], 8);
                        }else{
                          tmp_weight[0] = simd_getr(tmp_weight[0]);
                          tmp_weight[1] = simd_getr(tmp_weight[1]);
                          tmp_weight[2] = simd_getr(tmp_weight[2]);
                          tmp_weight[3] = simd_getr(tmp_weight[3]);
                        }

                        for(i = 0; i<4; ++i){
                          for(j = 0; j<4; ++j){
		             tmp_output[i*4+j] += tmp_input[j]*tmp_weight[i];
		          }
		        }
			local_input_ptr += B/8/SIMDSIZE;
			local_weight_ptr++;
                      }//cNi

		      for(i = 0 ; i < 4; ++i){
			for(j = 0; j < 4; ++j){
			  *local_output_ptr += tmp_output[i*4+j];
			  local_output_ptr++;
			}
			local_output_ptr += -4+B/8/SIMDSIZE;
		      }    
		      local_input_ptr  += -Ni/8*B/SIMDSIZE/8+4;
		      local_weight_ptr += -Ni/8;
		      local_output_ptr += -4*B/8/SIMDSIZE+4;
		    }//cB
		    local_input_ptr += -B/SIMDSIZE/8;
		    local_weight_ptr += 4*Ni/8;
		    local_output_ptr += 3*B/SIMDSIZE/8;
		  }//cNo
                }//ccCore end Calc
	#endif	
              }//if

            }//cKc
	    dma_wait(&replyget, 1); replyget = 0;
	    input_calc_index = 1-input_calc_index;
	    input_load_index = 1-input_load_index;
	    //input back inner
	    input_ptr += Ni*B;
	   
          }//cCi
	  //input forward inner
	  input_ptr -= Ni*B*(CiEnd - CoStart +1);
	  input_ptr += Ni*B*Ci;
      }//cKc

      //input back outer
      input_ptr -= Ni*B*Ci*K;
      
      jj=0;
      for(ii=CoStart; ii<CoEnd; ++ii){
          dma(dma_put_output, (long)(output_ptr), (long)(local_output+jj*B*No/64/SIMDSIZE));
          dma_wait(&replyput, 1); replyput = 0;
	  output_ptr += B*No;
          jj++;
      }
      //output_ptr += Ci*Ni*B;
      //input forward outer
      input_ptr += Ni*B*Ci;
    }//cRo

  }//CoStart


  ldm_free(local_input, sizeof(Type)*Ni*B/8/8*2);
  ldm_free(local_weight, sizeof(Type)*Ni*No/8/8*K);
  ldm_free(local_output, sizeof(Type)*No*B/8/8*CStride);

}//main func

