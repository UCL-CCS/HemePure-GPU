CXX=hipcc
CXXFLAGS=-O3 --offload-arch=gfx90a --gpu-max-threads-per-block=256

test: ./repro_test
	./repro_test

./repro_test: main.cu GPU_Collide_Stream_wall_sBB_Iolets.hpp precursor_data_7.h api.h
	$(CXX) -o ./repro_test $(CXXFLAGS) main.cu

clean:
	rm -f ./repro_test *.o
