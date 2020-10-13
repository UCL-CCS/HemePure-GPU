
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_NET_BUILDINFO_H_IN
#define HEMELB_NET_BUILDINFO_H_IN
namespace hemelb
{
  namespace net
  {
    typedef CoalescePointPoint PointPointImpl ;
    typedef SeparatedGathers GathersImpl ;
    typedef SeparatedAllToAll AllToAllImpl ;
/* #undef HEMELB_SEPARATE_CONCERNS */
    #ifdef HEMELB_SEPARATE_CONCERNS
    static const bool separate_communications = true;
    #else
    static const bool separate_communications = false;
    #endif
  }
}
#endif // HEMELB_NET_BUILDINFO_H_IN
