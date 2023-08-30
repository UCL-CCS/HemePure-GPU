CXX=hipcc
CXXFLAGS=-O3 -g -ggdb -I. --offload-arch=gfx90a

test: ./repro_test
	./repro_test

./repro_test: main.cu GPU_Collide_Stream_Iolets.hpp api.h nash_data_1.h
	$(CXX) -o ./repro_test $(CXXFLAGS) main.cu

clean:
	rm -f ./repro_test *.o out
