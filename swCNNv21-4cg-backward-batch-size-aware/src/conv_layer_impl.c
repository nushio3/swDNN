/*************************************************************************
	> File Name: ConvLayer_impl.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 10:25:15 AM CST
 ************************************************************************/

#include<stdio.h>
#include"conv_layer_impl.h"

//B, Ni, Ci, Ri
inline int inGetIdx(int cB, int cNi, int cCi, int cRi, int B, int Ni, int Ci){
  return cB + cNi*B + cCi*B*Ni + cRi*Ci*Ni*B;
}

//B, No, Co, Ro
inline int outGetIdx(int cB, int cNo, int cCo, int cRo, int B, int No, int Co){
  return cB + cNo*B + cCo*B*No + cRo*Co*No*B;
}

//Ni, No, K, K
inline int weightGetIdx(int cNi, int cKc, int cKr, int cNo, int Ni, int No, int K){
  return cNi + cNo*Ni + (cKr*K + cKc)*No*Ni;
}

void conv_forward_impl(Type* input, Type* weight, Type* output,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
  int cB,cNo,cNi,cRo,cCo,cKr,cKc;
  int Co = Ci-K+1;
  int Ro = Ri-K+1;
  printf("B: %d Ni: %d No: %d\n", B, Ni, No);

  for(cRo=0; cRo<Ro; cRo++)
    for(cCo=0; cCo<Co; cCo++){
      for(cNo=0; cNo<No; cNo++){
          for(cKr = 0 ;cKr<K; cKr++)
            for(cKc = 0; cKc<K; cKc++){
              for(cNi = 0; cNi<Ni; cNi++){
                for(cB = 0; cB<B; cB++){
                  *(output + outGetIdx(cB, cNo, cCo, cRo, B, No, Co)) += 
                    *(input + inGetIdx(cB, cNi, cCo+cKc, cRo+cKr, B, Ni, Ci)) * 
                    *(weight + weightGetIdx(cNi, cKc, cKr, cNo, Ni, No, K));
                    
                }
            }
          }
      }//cNo
    }//cCo
}

void conv_backward_impl(Type* in, Type* out_grad, Type* weight,
        Type* in_grad, Type* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)

{
    int cNi, cNo, cB, cCo, cRo, cCi, cRi, cKr, cKc;
    int Co = Ci-K+1;
    int Ro = Ri-K+1;
    int Pad = K-1;
    int gr, gc, lr, lc;

// in_grad = conv(out_grad, weight, 'full')
    for( cB = 0; cB < B; cB++ ){
        for( cNo = 0; cNo < No; cNo++ )    
            for( cNi = 0; cNi < Ni; cNi++ )    
                for( cCi = 0; cCi < Ci; cCi++)
                    for( cRi = 0; cRi < Ri; cRi++){
                        Type sum = 0.0;
                        for(cKr=0; cKr < K; ++cKr)
                            for(cKc=0; cKc < K; ++cKc){
                                gr = cKr+cRi;
                                gc = cKc+cCi;
                                lr = gr-Pad;
                                lc = gc-Pad;
                                if(lr >= 0 && lr < Ro && lc >= 0 && lc < Co){
                                    sum += *(out_grad+outGetIdx(cB, cNo, lc, lr, B, No, Co)) * 
                                      *(weight+weightGetIdx(cNi, cKc, cKr,cNo,  Ni, No, K)); 
                                }
                            }//cKc
                        *(in_grad+inGetIdx(cB, cNi, cCi, cRi, B, Ni, Ci)) += sum;
                    }
    }//cB
    
// weight_diff = conv(in, out_grad, 'valid')
    for(cB = 0; cB<B; cB++)
      for(cNo=0; cNo<No; cNo++)
        for(cNi = 0; cNi<Ni; cNi++)
          for(cKr = 0 ;cKr<K; cKr++)
		        for(cKc = 0; cKc<K; cKc++)
              for(cRo=0; cRo<Ro; cRo++)
                for(cCo=0; cCo<Co; cCo++)
                {
                  *(weight_diff+ weightGetIdx(cNi, cKc, cKr, cNo,  Ni, No, K)) += 
                    *(in + inGetIdx(cB, cNi, cCo+cKc, cRo+cKr, B, Ni, Ci)) * 
                    *(out_grad + outGetIdx(cB, cNo, cCo, cRo, B, No, Co));
                }
}

