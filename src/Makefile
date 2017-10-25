# NVCC := $(shell nvcc --version > /dev/null 2>&1; echo $$?)
FOMP := $(shell $(CXX) -fopenmp -c conv.cpp -o /dev/null > /dev/null 2>&1; echo $$?)
CFLAGS := -O3 -Wall -fPIC
OBJ := conv.o

ifeq ($(FOMP),0)
  CFLAGS := $(CFLAGS) -fopenmp
endif


libnnlib.so: $(OBJ)
	$(CXX) --shared $(CFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

clean:
	-rm *.o *.cu libknet8.so

.PRECIOUS: %.cu
