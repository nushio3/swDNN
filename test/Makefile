MCC = swcc 
CC = sw5cc.new
LCC = sw5f90
all: example 
dots: core_functions.S

CFLAGS = -O3 -msimd 
NOCFLAGS = -O0 -msimd
LIBFLAGS = -hybrid -lslave -allshare -lm 

SlaveOBJS= swConvForward.o swConvAddBias.o swZeroPad.o swReluForward.o swReluBackward.o 
HostOBJS = def.o netconfig.o softmax.o ConvForward.o ReLU.o mlp.o

def.o:def.c def.h
	$(CC) $(CFLAGS) -host -c $< -o $@
netconfig.o: netconfig.c
	$(CC) $(CFLAGS) -host -c $< -o $@
softmax.o:softmax.c
	$(CC) $(CFLAGS) -host -c $< -o $@
mlp.o:mlp.c
	$(CC) $(CFLAGS) -host -c $< -o $@
ReLU.o:ReLU.c
	$(CC) $(CFLAGS) -host -c $< -o $@
ConvForward.o: ConvForward.c 
	$(CC) $(CFLAGS) -host -c $< -o $@
test.o: newtest.c
	$(CC) $(CFLAGS) -host -c $< -o $@

#test.o:test.c
#	$(CC) $(CFLAGS) -host -c $< -o $@

swReluForward.o:swReluForward.c
	$(CC) $(CFLAGS) -slave -c $< -o $@
swReluBackward.o:swReluBackward.c
	$(CC) $(CFLAGS) -slave -c $< -o $@
swConvForward.o:swConvForward.S
	$(CC) $(CFLAGS) -slave -c $< -o $@
swZeroPad.o:swZeroPad.c
	$(CC) $(CFLAGS) -slave -c $< -o $@
swConvAddBias.o:swConvAddBias.c
	$(CC) $(CFLAGS) -slave -c $< -o $@

convf:test.o $(SlaveOBJS) $(HostOBJS)
	$(LCC) $(LIBFLAGS) $^ ./libswblas_opt.a  -o test 

run_convf:
	time bsub -b -I -m 1 -p -q q_sw_qh -host_stack 2048 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 16000 -n 1 -o run.log -cgsp 64 ./test
	
clean:
	-rm test *.o
