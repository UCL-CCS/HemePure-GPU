cmake -DHEMELB_GPU_BACKEND=CUDA \
		 -DCMAKE_CXX_COMPILER=mpicxx \
		 -DCMAKE_C_COMPILER=mpicc \
		 -DHEMELB_COMPUTE_ARCHITECTURE=NEUTRAL \
		 -DCMAKE_CXX_EXTENSIONS=OFF \
		 -DHEMELB_USE_SSE3=OFF \
		 -DCMAKE_CUDA_ARCHITECTURES=70 \
		 -DCTEMPLATE_LIBRARY=/gpfs/alpine/proj-shared/stf006/bjoo/HemeLBNew/HemePure_GPU_new-summit/dep/install/lib/libctemplate.a \
			..

#  -DAMDGPU_TARGETS="gfx90a"\
#	 -DGPU_TARGETS="gfx90a" \
