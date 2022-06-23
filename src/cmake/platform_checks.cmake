include(CheckCXXSourceCompiles)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_TIRPC libtirpc)

find_path(TIRPC_INCLUDE_DIRS
	  NAMES netconfig.h
	  PATH_SUFFIXES tirpc
	  HINTS ${PC_TIRPC_INCLUDE_DIRS}
	  )

find_library(TIRPC_LIBRARIES
	     NAMES tirpc
	     HINTS ${PC_TIRPC_LIBRARY_DIRS}
	     )

set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES} ${TIRPC_INCLUDE_DIRS}")
set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES} ${TIRPC_LIBRARIES}")

include_directories("${TIRPC_INCLUDE_DIRS}")
list(APPEND HEMELB_LIBRARIES ${TIRPC_LIBRARIES})

CHECK_CXX_SOURCE_COMPILES("#include <sys/time.h>\n#include <sys/resource.h>\nint main(int c,char** v){ rusage usage;\ngetrusage(RUSAGE_SELF, &usage);\nreturn usage.ru_maxrss; }" HAVE_RUSAGE)
CHECK_CXX_SOURCE_COMPILES("
#include <stdint.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
int main(int count, char** v){
	char buffer[15] = \"aaaaaaaaaaaaa\";
	XDR xdr;
	xdrmem_create(&xdr, buffer, 32, XDR_ENCODE);
	uint16_t a;
	uint32_t b;
	uint64_t c;
	xdr_uint16_t(&xdr, &a);
	xdr_uint32_t(&xdr, &b);
	xdr_uint64_t(&xdr, &c);
	return b;
}" HAVE_XDRUINTXX_T)

# cstdint is the c++11 version of C99 stdint.h.
# better to go with cstdint, but stdint.h is available more widely.
find_path(HAVE_STDINT_H stdint.h)
find_path(HAVE_CSTDINT cstdint)
if(HAVE_CSTDINT)
	add_definitions(-DHEMELB_HAVE_CSTDINT=TRUE)
elseif(NOT HAVE_STDINT_H)
	message(ERROR "Neither cstdint nor stdint.h found")
endif()
