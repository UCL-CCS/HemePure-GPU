/*
Copyright 2018 Rene Halver, Forschungszentrum Juelich GmbH, Germany
Copyright 2018 Godehard Sutmann, Forschungszentrum Juelich GmbH, Germany

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
   list of conditions and the following disclaimer in the documentation and/or 
   other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may 
   be used to endorse or promote products derived from this software without 
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ALL_CUSTOM_EXCEPTIONS_INC
#define ALL_CUSTOM_EXCEPTIONS_INC

#include <exception>
#include <string>
#include <sstream>

// customized exception for ALL, modified for each specific exception type
struct ALL_Custom_Exception : public std::exception
{
    protected:
        // name of the exception
        const char* loc_desc;
        // function the exception occured in
        const char* func;
        // file the exception occured in
        const char* file;
        // information on the exception
        const char* info;
        // line the exception occured in
        int line;
        // error message
        std::string error_msg;
    public:
    ALL_Custom_Exception(const char* file_ = "",
                         const char* f_ = "", 
                         int l_ = -1, 
                         const char* i_ = "", 
                         const char* loc_desc_ = "ALLCustomException") :
        file(file_),
        func(f_),
        line(l_),
        info(i_),
        loc_desc(loc_desc_) {
                                std::stringstream ss;
                                ss << loc_desc << ": " << info << "\n" 
                                                      << "Function: " << func << "\n" 
                                                      << "File: " << file << "\n"
                                                      << "Line: " << line << "\n";
                                error_msg = ss.str();
                            }
    const char* get_func() const {return func;}
    int get_line() const { return line; }
    const char* get_info() { return info; }

    virtual const char* what () const throw()
    {
        return error_msg.c_str();
    }
};


// Execption to be used for missmatches in dimension for Point class
struct ALL_Point_Dimension_Missmatch_Exception : public ALL_Custom_Exception
{
    public:
    ALL_Point_Dimension_Missmatch_Exception(const char* file_,
                                            const char* f_, 
                                            int l_, 
                                            const char* i_ = "Dimension missmatch in Point objects.",
                                            const char* loc_desc_ = "ALLPointDimMissmatchException") :
        ALL_Custom_Exception(file_,f_,l_,i_,loc_desc_) {}
};

// Execption to be used for invalid Communicators in for load-balancing classes
struct ALL_Invalid_Comm_Type_Exception : public ALL_Custom_Exception
{
    public:
    ALL_Invalid_Comm_Type_Exception(const char* file_,
                                    const char* f_, 
                                    int l_, 
                                    const char* i_ = "Type of MPI communicator not valid.",
                                    const char* loc_desc_ = "ALLCommTypeInvalidException") :
    ALL_Custom_Exception(file_,f_,l_,i_,loc_desc_) {}
};

// Execption to be used for invalid parameters in any type of classes
struct ALL_Invalid_Argument_Exception : public ALL_Custom_Exception
{
    public:
    ALL_Invalid_Argument_Exception(const char* file_,
                                   const char* f_ = "", 
                                   int l_ = -1, 
                                   const char* i_ = "Invalid argument given.",
                                   const char* loc_desc_ = "ALLInvalidArgumentException") :
    ALL_Custom_Exception(file_,f_,l_,i_,loc_desc_) {}
};

// Execption to be used for invalid parameters in any type of classes
struct ALL_Internal_Error_Exception : public ALL_Custom_Exception
{
    public:
    ALL_Internal_Error_Exception(const char* file_,
                                 const char* f_ = "", 
                                 int l_ = -1, 
                                 const char* i_ = "Internal error occured, see description.",
                                 const char* loc_desc_ = "ALLInternalErrorException") :
    ALL_Custom_Exception(file_,f_,l_,i_,loc_desc_) {}
};


#endif
