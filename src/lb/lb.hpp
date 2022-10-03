#ifndef HEMELB_LB_LB_HPP_SWITCH
#define HEMELB_LB_LB_HPP_SWITCH

#ifdef HEMELB_USE_HIP
#include "lb_hip.hpp"
#else
#include "lb_cuda.hpp"
#endif

#endif /* HEMELB_LB_LB_HPP_SWITCH */
