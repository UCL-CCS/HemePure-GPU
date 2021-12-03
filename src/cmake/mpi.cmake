# MPI
# ---
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${CMAKE_CXX_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})

# does MPI implementation have a const-correct API (supports MPI 3)
set(CMAKE_REQUIRED_FLAGS -Werror)
set(CMAKE_REQUIRED_DEFINITIONS ${MPI_COMPILE_FLAGS})
set(CMAKE_REQUIRED_INCLUDES ${MPI_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${MPI_LIBRARIES})
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
