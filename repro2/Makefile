CXX=clang++
CXXFLAGS=-O2 -std=c++17 -fsycl -fsycl-unnamed-lambda -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a

./repro_test: main.cu GPU_Collide_Stream_wall_sBB_Iolets.cu GPU_Collide_Stream_wall_sBB_Iolets.hpp  precursor_data_7.h api.h
	$(CXX) -o ./repro_test $(CXXFLAGS) -x c++ main.cu

clean:
	rm -f ./repro_test *.o
