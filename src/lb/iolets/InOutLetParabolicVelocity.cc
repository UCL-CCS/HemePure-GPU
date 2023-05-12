
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.
#include "lb/iolets/InOutLetParabolicVelocity.h"
#include "configuration/SimConfig.h"

namespace hemelb
{
  namespace lb
  {
    namespace iolets
    {
      InOutLetParabolicVelocity::InOutLetParabolicVelocity() :
          maxSpeed(0.), warmUpLength(0)
      {
      }

      InOutLetParabolicVelocity::~InOutLetParabolicVelocity()
      {
      }

      InOutLet* InOutLetParabolicVelocity::Clone() const
      {
        InOutLet* copy = new InOutLetParabolicVelocity(*this);
        return copy;
      }

      LatticeVelocity InOutLetParabolicVelocity::GetVelocity(const LatticePosition& x,
                                                             const LatticeTimeStep t) const
      {

        //printf("Entering GetVelocity function in InOutlet Parabolic Velocity \n\n");
        // v(r) = vMax (1 - r**2 / a**2)
        // where r is the distance from the centreline
        LatticePosition displ = x - position;
        LatticeDistance z = displ.Dot(normal);
        Dimensionless rSq = (displ.GetMagnitudeSquared() - z * z) / (radius * radius);

        // Get the max velocity
        LatticeSpeed max = maxSpeed;
        //printf("Max speed = %.5f \n\n", maxSpeed);

        // If we're in the warm-up phase, scale down the imposed velocity
        if (t < warmUpLength)
        {
          max *= t / double(warmUpLength);
        }
        //printf("Max speed = %.5f \n\n", max);

        //printf("Returned Value.x : %.5f, Value.y : %.5f, Value.z : %.5f \n\n", normal.x * (max * (1. - rSq)), normal.y * (max * (1. - rSq)), normal.z * (max * (1. - rSq)) );

        // Brackets to ensure that the scalar multiplies are done before vector * scalar.
        return normal * (max * (1. - rSq));
      }

      // April 2023 - IZ
      LatticeVelocity InOutLetParabolicVelocity::GetVelocity_prefactor(const LatticePosition& x,
                                                             const LatticeTimeStep t) const
      {
        //printf("Entering GetVelocity_prefactor function in InOutlet Parabolic Velocity \n\n");=
        return normal * 0.0;
      }
      //

    }
  }
}
