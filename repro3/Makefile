CXX=clang++
CXXFLAGS=-O3 -std=c++17 -fsycl -fsycl-unnamed-lambda -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a  -x c++

#CXX=hipcc
#CXXFLAGS=-std=c++17 -O1 --offload-arch=gfx90a
test: ./repro_test
	./repro_test

./repro_test: main.cu GPU_Collide_Stream_Iolets.hpp repro3_data_inlet2.h api.h
	$(CXX) -o ./repro_test $(CXXFLAGS) main.cu

clean:
	rm -f ./repro_test *.o
