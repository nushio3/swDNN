/*************************************************************************
	> File Name: Tensor.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Thu 29 Dec 2016 07:42:19 PM CST
 ************************************************************************/

#include"util.h"
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<memory.h>
#include<ctime>

class tensor{
public:
    Type* data;
    int n1, n2, n3, n4;
    int size;

    tensor(int _n1, int _n2, int _n3):
        n1(_n1),n2(_n2),n3(_n3),n4(0)
    {
        size = n1*n2*n3;
        data = (Type*)malloc(sizeof(Type)*n1*n2*n3);
        memset(data, 0.0, sizeof(Type)*size);
    }
    tensor(int _n1, int _n2, int _n3, int _n4):
        n1(_n1),n2(_n2),n3(_n3),n4(_n4)
    {
        size = n1 * n2 * n3 * n4;
        data = (Type*)malloc(sizeof(Type)*n1*n2*n3*n4);
        memset(data, 0.0, sizeof(Type)*size);
    }
    tensor(const tensor& T){
        n1 = T.n1; n2 = T.n2;
        n3 = T.n3; n4 = T.n4;
        size = T.size;
        data = (Type*)malloc(sizeof(Type)*size);
        memcpy(data, T.data, sizeof(Type)*size);
    }

    void rand_init()
    {
      srand(unsigned(time(0)));
      for(int i=0; i < size; ++i){
        data[i] = (double)rand()/RAND_MAX;
      }
    }

    void load_tensor(const char* filename){
      FILE* fh = fopen(filename, "r");
      if(!fh)
        printf("[ERROR] no such file %s\n", filename);
      else{
        for(int i=0; i < size; ++i)
          fscanf(fh, "%lf", data+i);
      }
      fclose(fh);
    }

    void store_tensor(const char* filename){
      FILE* fh = fopen(filename, "w");
      if(!fh)
        printf("[ERROR] no such file %s\n", filename);
      else{
        for(int i=0; i < size; ++i)
          fprintf(fh, "%lf\n", *(data+i));
      }
      fclose(fh);
    }


    bool compare(tensor T){
      if(size != T.size)
        return false;
      else{
        for(int i=0; i<size; ++i){
          if(fabs(data[i]-T.data[i]) > 1e-6)
            return false;
        }
      }
      return true;
    }
};
