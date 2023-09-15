#pragma once

#include "cuda_kernels_def_decl/sycl/stream_manager.h"

namespace hemelb {
  namespace GPU {

	using Stream_t = hemelb::GPU::Impl::StreamManager::Stream_t;

  }
}

