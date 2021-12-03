
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "configuration/CommandLine.h"
#include <cstring>
#include <cstdlib>
namespace hemelb
{
  namespace configuration
  {

    CommandLine::CommandLine(int aargc, const char * const * const aargv) :
      inputFile("input.xml"), outputDir(""), images(10), argc(aargc),
          argv(aargv)
    {

      // There should be an odd number of arguments since the parameters occur in pairs.
      if ( (argc % 2) == 0)
      {
        throw OptionError() << "There should be an odd number of arguments since the parameters occur in pairs.";
      }

      // All arguments are parsed in pairs, one is a "-<paramName>" type, and one
      // is the <parametervalue>.
      for (int ii = 1; ii < argc; ii += 2)
      {
        const char* const paramName = argv[ii];
        const char* const paramValue = argv[ii + 1];
        if (std::strcmp(paramName, "-in") == 0)
        {
          inputFile = std::string(paramValue);
        }
        else if (std::strcmp(paramName, "-out") == 0)
        {
          outputDir = std::string(paramValue);
        }
        else if (std::strcmp(paramName, "-i") == 0)
        {
          char *dummy;
          images = (unsigned int) (strtoul(paramValue, &dummy, 10));
        }
        else
        {
          throw OptionError() << "Unknown option: " << paramName;
        }
      }
    }

    std::string CommandLine::GetUsage()
    {
      std::string ans("Correct usage: hemelb [-<Parameter Name> <Parameter Value>]* \n");
      ans.append("Parameter name and significance:\n");
      ans.append("-in \t path to the configuration xml file (default is config.xml)\n");
      ans.append("-out \t path to the output folder (default is based on input file, e.g. config_xml_results)\n");
      ans.append("-i \t number of images to create (default is 10)\n");
      return ans;
    }
  }
}
