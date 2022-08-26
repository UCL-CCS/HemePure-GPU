# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

set(hemelb_extraction_sources
	GeometrySelector.cc
	StraightLineGeometrySelector.cc
	LocalPropertyOutput.cc
	IterableDataSource.cc
	PlaneGeometrySelector.cc
	PropertyWriter.cc
	InletSelector.cc
	WholeGeometrySelector.cc
	GeometrySurfaceSelector.cc
	SurfacePointSelector.cc)

set(hemelb_extraction_hip_sources
	PropertyActor.cc
	LbDataSourceIterator.cc
)

set_source_files_properties(${hemelb_extraction_hip_sources}
	PROPERTIES LANGUAGE HIP)

add_library(hemelb_extraction
	${hemelb_extraction_sources}
	${hemelb_extraction_hip_sources}
)

