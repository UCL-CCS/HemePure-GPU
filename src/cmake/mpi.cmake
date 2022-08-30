# MPI
# ---
# does MPI implementation have a const-correct API (supports MPI 3)
# New Cmake Style: Just add MPI as an imported target to libraries
# Using MPI::MPI_CXX
#
find_package(MPI COMPONENTS CXX REQUIRED)

set(CMAKE_REQUIRED_FLAGS -Werror)
set(CMAKE_REQUIRED_DEFINITIONS ${MPI_CXX_COMPILE_FLAGS})
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
CHECK_CXX_SOURCE_COMPILES("#include <mpi.h>
int main(int argc, char* argv[]) {
	const int send = 0;
	int recv;
	MPI_Request req;
	MPI_Init(&argc, &argv);
	MPI_Irecv(&recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
	MPI_Send(&send, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Finalize();
}" HAVE_CONSTCORRECTMPI)
