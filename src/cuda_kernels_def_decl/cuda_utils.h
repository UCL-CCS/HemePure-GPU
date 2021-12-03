/**
 * From https://bitbucket.org/rvuduc/volkov-gtc10/src/master/cuda_utils.h 
 *
 * CUDA Utilities
 *
 *  These functions are similar to the CUDA_SAFE_CALL
 *  from the CUDA SDK as well as book.h from 
 *
 *  http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
 *
 *
 **/


#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cassert>
#include <iostream>

extern void cudaCheck__ (cudaError_t err, const char *file, size_t line);

#define cudaCheck(err) cudaCheck__ ((err), __FILE__, __LINE__)

#define CHECK_NULL( a ) {if (a == NULL) { \
                            printf( "CHECK_ERROR failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#endif  // __CUDA_UTILS_H__
