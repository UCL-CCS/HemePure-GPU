
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_REPORTING_BUILDINFO_H_IN
#define HEMELB_REPORTING_BUILDINFO_H_IN
#include "reporting/Reportable.h"
namespace hemelb
{
  namespace reporting
  {
    static const std::string build_type="";
    static const std::string optimisation="-O3";
    static const std::string use_sse3="OFF";
    static const std::string build_time="";
    static const std::string reading_group_size="2";
    static const std::string lattice_type="D3Q19";
    static const std::string kernel_type="LBGK";
    static const std::string wall_boundary_condition="SIMPLEBOUNCEBACK";
    static const std::string inlet_boundary_condition="NASHZEROTHORDERPRESSUREIOLET";
    static const std::string outlet_boundary_condition="NASHZEROTHORDERPRESSUREIOLET";
    static const std::string wall_inlet_boundary_condition="NASHZEROTHORDERPRESSURESBB";
    static const std::string wall_outlet_boundary_condition="NASHZEROTHORDERPRESSURESBB";
    static const std::string point_point_impl="Coalesce";
    static const std::string gathers_impl="Separated";
    static const std::string alltoall_impl="Separated";
    static const std::string separate_concerns="OFF";


    class BuildInfo : public Reportable {
      void Report(ctemplate::TemplateDictionary& dictionary){
        ctemplate::TemplateDictionary *build = dictionary.AddSectionDictionary("BUILD");
        build->SetValue("TYPE", build_type);
        build->SetValue("OPTIMISATION", optimisation);
        build->SetValue("USE_SSE3", use_sse3);
        build->SetValue("TIME", build_time);
        build->SetValue("READING_GROUP_SIZE", reading_group_size);
        build->SetValue("LATTICE_TYPE", lattice_type);
        build->SetValue("KERNEL_TYPE", kernel_type);
        build->SetValue("WALL_BOUNDARY_CONDITION", wall_boundary_condition);
        build->SetValue("INLET_BOUNDARY_CONDITION", inlet_boundary_condition);
        build->SetValue("OUTLET_BOUNDARY_CONDITION", outlet_boundary_condition);
        build->SetValue("WALL_INLET_BOUNDARY_CONDITION", wall_inlet_boundary_condition);
        build->SetValue("WALL_OUTLET_BOUNDARY_CONDITION", wall_outlet_boundary_condition);
        build->SetValue("SEPARATE_CONCERNS",separate_concerns);
        build->SetValue("ALLTOALL_IMPLEMENTATION",alltoall_impl);
        build->SetValue("GATHERS_IMPLEMENTATION",gathers_impl);
        build->SetValue("POINTPOINT_IMPLEMENTATION",point_point_impl);
      }
    };
  }
}
#endif // HEMELB_REPORTING_BUILDINFO_H_IN
