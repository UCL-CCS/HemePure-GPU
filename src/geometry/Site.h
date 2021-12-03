
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_GEOMETRY_SITE_H
#define HEMELB_GEOMETRY_SITE_H

#include "units.h"
#include "geometry/SiteData.h"
#include "util/Vector3D.h"

namespace hemelb
{
  namespace geometry
  {

    template<class DataSource>
    class Site
    {
      public:
        Site(site_t localContiguousIndex, DataSource &latticeData) :
            index(localContiguousIndex), latticeData(latticeData)
        {
        }

        inline bool IsWall() const
        {
          return GetSiteData().IsWall();
        }

        inline bool IsSolid() const
        {
          return GetSiteData().IsSolid();
        }

        inline unsigned GetCollisionType() const
        {
          return GetSiteData().GetCollisionType();
        }

        inline SiteType GetSiteType() const
        {
          return GetSiteData().GetSiteType();
        }

        inline int GetIoletId() const
        {
          return GetSiteData().GetIoletId();
        }

        inline bool HasWall(Direction direction) const
        {
          return GetSiteData().HasWall(direction);
        }

        inline bool HasIolet(Direction direction) const
        {
          return GetSiteData().HasIolet(direction);
        }

        template<typename LatticeType>
        inline distribn_t GetWallDistance(Direction direction) const
        {
          return latticeData.template GetCutDistance<LatticeType>(index, direction);
        }

        inline distribn_t* GetWallDistances()
        {
          return latticeData.GetCutDistances(index);
        }

        inline const distribn_t* GetWallDistances() const
        {
          return latticeData.GetCutDistances(index);
        }

        inline const util::Vector3D<distribn_t>& GetWallNormal() const
        {
          return latticeData.GetNormalToWall(index);
        }

        inline util::Vector3D<distribn_t>& GetWallNormal()
        {
          return latticeData.GetNormalToWall(index);
        }

        const LatticeForceVector& GetForce() const
        {
          return latticeData.GetForceAtSite(index);
        }

        void SetForce(LatticeForceVector const &_force)
        {
          return latticeData.SetForceAtSite(index, _force);
        }

        void AddToForce(LatticeForceVector const &_force)
        {
          return latticeData.AddToForceAtSite(index, _force);
        }

        inline site_t GetIndex() const
        {
          return index;
        }

        /**
         * This returns the index of the distribution to stream to. If streaming would take the
         * distribution out of the geometry, we instead stream to the 'rubbish site', an extra
         * position in the array that doesn't correspond to any site in the geometry.
         *
         * @param direction
         * @return
         */
        template<typename LatticeType>
        inline site_t GetStreamedIndex(Direction direction) const
        {
          return latticeData.template GetStreamedIndex<LatticeType>(index, direction);
        }

        template<typename LatticeType>
        inline const distribn_t* GetFOld() const
        {
          return latticeData.GetFOld(index * LatticeType::NUMVECTORS);
        }

        // Non-templated version of GetFOld, for when you haven't got a lattice type handy
        inline const distribn_t* GetFOld(int numvectors) const
        {
          return latticeData.GetFOld(index * numvectors);
        }

        inline const SiteData& GetSiteData() const
        {
          return latticeData.GetSiteData(index);
        }

        inline SiteData& GetSiteData()
        {
          return latticeData.GetSiteData(index);
        }

        inline const util::Vector3D<site_t>& GetGlobalSiteCoords() const
        {
          return latticeData.GetGlobalSiteCoords(index);
        }

      protected:
        site_t index;
        DataSource & latticeData;
    };
  }
}

#endif
