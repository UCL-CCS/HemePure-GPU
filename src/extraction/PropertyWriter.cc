
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "extraction/PropertyWriter.h"

namespace hemelb
{
  namespace extraction
  {
    PropertyWriter::PropertyWriter(IterableDataSource& dataSource,
                                   const std::vector<PropertyOutputFile*>& propertyOutputs,
                                   const net::IOCommunicator& ioComms)
    {
      //printf("Size of propertyOutputs: %d \n\n", propertyOutputs.size());
      for (unsigned outputNumber = 0; outputNumber < propertyOutputs.size(); ++outputNumber)
      {
        localPropertyOutputs.push_back(new LocalPropertyOutput(dataSource, propertyOutputs[outputNumber], ioComms));
      }
    }

    PropertyWriter::~PropertyWriter()
    {
      for (unsigned outputNumber = 0; outputNumber < localPropertyOutputs.size(); ++outputNumber)
      {
        delete localPropertyOutputs[outputNumber];
      }
    }

    const std::vector<LocalPropertyOutput*>& PropertyWriter::GetPropertyOutputs() const
    {
      return localPropertyOutputs;
    }

    void PropertyWriter::Write(unsigned long iterationNumber, unsigned long initial_iterationNumber, unsigned long max_iterationNumber) const
    {
      for (unsigned outputNumber = 0; outputNumber < localPropertyOutputs.size(); ++outputNumber)
      {
        localPropertyOutputs[outputNumber]->Write((uint64_t) iterationNumber, (uint64_t) initial_iterationNumber, (uint64_t) max_iterationNumber);
      }
    }
  }
}
