// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_EXTRACTION_ASYNCH_WRITE_H
#define HEMELB_EXTRACTION_ASYNCH_WRITE_H

// Added 18 July 2020 - Multithreading
//#include <pthread.h>
//#include <thread>
#include "cuda_kernels_def_decl/Threads.h"
//#include "extraction/PropertyActor.h"

Threads Worker;

int ThreadWork_Save_Files(Threads::Thread* thread);




#endif /* HEMELB_EXTRACTION_ASYNCH_WRITE_H */
