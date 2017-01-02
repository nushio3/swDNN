CC = sw5cc.new
CXX = sw5CC
LD = mpiCC 
all: test

CFLAGS = -O3 -I../include 
NOCFLAGS = -O0 
LIBFLAGS = 

OBJ = ../build/test.o ../build/conv_layer_impl.o ../build/sw_conv_layer_impl.o \
	  ../build/sw_slave_conv_valid.o ../build/gemm_asm.o ../build/sw_slave_conv_full.o

../build/test.o: test.cpp
	$(CXX) $(CFLAGS) -host -c $< -o $@
../build/conv_layer_impl.o: conv_layer_impl.c
	$(CC) $(CFLAGS) -host -c $< -o $@
../build/sw_conv_layer_impl.o: sw_conv_layer_impl.c
	$(CC) $(CFLAGS) -host -c $< -o $@
../build/sw_slave_conv_valid.o: sw_slave_conv_valid.c
	$(CC) $(CFLAGS) -slave -msimd -c $< -o $@
../build/sw_slave_conv_full.o: sw_slave_conv_full.c
	$(CC) $(CFLAGS) -slave -msimd -c $< -o $@
../build/gemm_asm.o: gemm.S
	$(CC) $(CFLAGS) -slave -msimd -c $< -o $@
test: $(OBJ)
	$(LD) -hybrid -lslave $(OBJ) -o test
	#$(LD) $(LIBFLAGS) $(OBJ) -o test
run:
	time bsub -b -I -m 1 -p -q q_sw_yfb -host_stack 1024 -share_size 6000 -n 1 -cgsp 64 ./test
clean:
	-rm test ../build/*.o


