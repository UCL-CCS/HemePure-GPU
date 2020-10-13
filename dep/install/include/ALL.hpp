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

#ifndef ALL_MAIN_HEADER_INC
#define ALL_MAIN_HEADER_INC

#include "ALL_Point.hpp"
#include "ALL_CustomExceptions.hpp"
#include "ALL_Tensor.hpp"
#ifdef ALL_VORONOI
#include "ALL_Voronoi.hpp"
#endif
#include "ALL_Staggered.hpp"
#include "ALL_Unstructured.hpp"
#include "ALL_Histogram.hpp"
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <iomanip>

#ifdef ALL_VTK_OUTPUT
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVoxel.h>
#include <vtkPolyhedron.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkMPIController.h>
#include <vtkProgrammableFilter.h>
#include <vtkInformation.h>
#endif

#define ALL_ESTIMATION_LIMIT_DEFAULT 0

enum ALL_LB_t : int 
            {
                // staggered grid load balancing
                STAGGERED = 0, 
                // tensor based load balancing
                TENSOR = 1, 
                // unstructured-mesh load balancing
                UNSTRUCTURED = 2,
                // voronoi cell based load balancing
                VORONOI = 3,
                // histogram based load balancing
                HISTOGRAM = 4 
            }; 

// T = data type for vertices, W data type for work
template <class T, class W> class ALL
{
    public:

        // constructor (empty)
        ALL() : work_array(NULL), 
#ifdef ALL_VTK_OUTPUT
                vtk_init(false),
#endif        
                balancer(NULL),
                loadbalancing_step(0),
                dimension(3) 
                {
                    // determine correct MPI data type for template T
                    if (std::is_same<T,double>::value) mpi_data_type_T = MPI_DOUBLE;
                    else if (std::is_same<T,float>::value) mpi_data_type_T = MPI_FLOAT;
                    else if (std::is_same<T,int>::value) mpi_data_type_T = MPI_INT;
                    else if (std::is_same<T,long>::value) mpi_data_type_T = MPI_LONG;

                    // determine correct MPI data type for template W 
                    if (std::is_same<W,double>::value) mpi_data_type_W = MPI_DOUBLE;
                    else if (std::is_same<W,float>::value) mpi_data_type_W = MPI_FLOAT;
                    else if (std::is_same<W,int>::value) mpi_data_type_W = MPI_INT;
                    else if (std::is_same<W,long>::value) mpi_data_type_W = MPI_LONG;
                }

        // constructor (dimension only)
        ALL(const int d_, T g) : ALL()       { 
                                                        gamma = g;
                                                        dimension = d_;
                                                        outline = new std::vector<T>(2*d_); 
                                                        estimation_limit = ALL_ESTIMATION_LIMIT_DEFAULT;
                                             }

        // constructor (dimension + vertices)
        ALL(const int d_, const std::vector<ALL_Point<T>>& inp, T g) : ALL(d_,g)
        {
            vertices = inp;
            estimation_limit = ALL_ESTIMATION_LIMIT_DEFAULT;
            calculate_outline();
        }

        // destructor
        ~ALL();

        // overwrite vertices
        // with a vector
        void set_vertices(std::vector<ALL_Point<T>>&);
        // with an array of T values (unsafe due to missing possibility to check T array)
        void set_vertices(const int n, const int dim, const T*);

        // set up work of domain

        // set up MPI communicator
        void set_communicator(MPI_Comm);
        
        // set gamma (correction value)
        void set_gamma(T g) { gamma = g;};

        // single value (staggered grid / tensor-based)
        void set_work(const W);

        // vector of work (cell based -> work per cell)
        void set_work(const std::vector<W>&);

        // separate method used to create the balancer object (no setup method included)
        void create_balancer(ALL_LB_t);

        // setup balancer
        void setup(ALL_LB_t);

        // calculate load-balance
        void balance(ALL_LB_t method, bool internal = false);

        // separate call ONLY for the setup method of the balancer
        void setup_balancer(ALL_LB_t method);

        // cleanup balancer
        void cleanup(ALL_LB_t);

        // getter functions

        // vertices 
        std::vector<ALL_Point<T>>& get_vertices() {return vertices;};
        std::vector<ALL_Point<T>>& get_result_vertices() {return result_vertices;};

        // dimension
        int get_dimension() {return dimension;}

        // work
        void get_work(std::vector<W>&);
        void get_work(W&);

        // neighbors
        int get_n_neighbors(ALL_LB_t);
        // provide list of neighbors (ranks)
        void get_neighbors(ALL_LB_t,std::vector<int>&);
        // provide a list of neighbor vertices (VORONOI only)
        void get_neighbor_vertices(ALL_LB_t,std::vector<T>&);
        // provide list of neighbors in each direction
        void get_neighbors(ALL_LB_t,int**);

        // set system size
        // (x_min, x_max, y_min, y_max, z_min, z_max)
        void set_sys_size(ALL_LB_t,std::vector<T>&);

        // method to set method specific data, without
        // the necessity to create dummy hooks for other
        // methods
        void set_method_data(ALL_LB_t,void*);

#ifdef ALL_VTK_OUTPUT        
        // print outlines and points within domain
        void print_vtk_outlines(int);
        // print vertices 
        void print_vtk_vertices(int);
#endif        

    private:
        // outer cuboid encasing the domain (for cuboid domains identical to domain)
        // defined by front left lower [0][*] and back upper right points [1][*]
        // where * is 0,...,dim-1
        std::vector<T>* outline;

        // gamma = correction factor
        T gamma;

        // dimension of the system (>=1)
        int dimension;

        // balancer
        void *balancer;

        // data containing cells (TODO: defintion of data layout, etc.)
        std::vector<W> *work_array;

        // number of vertices (for non-cuboid domains)
        int n_vertices;

        // number of neighbors
        int n_neighbors;

        // list of vertices
        std::vector<ALL_Point<T>> vertices;

        // list of vertices (after a balancing shift)
        std::vector<ALL_Point<T>> result_vertices;

        // list of neighbors
        std::vector<int> neighbors;

        // global MPI communicator
        MPI_Comm communicator;

        // calculate the outline of the domain using the vertices
        void calculate_outline();

        // limit for the estimation procedure
        int estimation_limit;

        // type for MPI communication
        MPI_Datatype mpi_data_type_T;
        MPI_Datatype mpi_data_type_W;

        int loadbalancing_step;

#ifdef ALL_VTK_OUTPUT
        bool vtk_init;
        vtkMPIController* controller;
        vtkMPIController* controller_vertices;
#endif        
};

template <class T, class W> ALL<T,W>::~ALL()
{
    if (outline)    delete outline;
    if (work_array) delete work_array;
}

template <class T, class W> void ALL<T,W>::set_vertices(std::vector<ALL_Point<T>>& inp)
{
    int dim = inp.at(0).get_dimension();
    if (dim != dimension) throw ALL_Point_Dimension_Missmatch_Exception(
                                __FILE__,
                                __func__,
                                __LINE__,
                                "Dimension of ALL_Points in input vector do not match dimension of ALL object.");
    // clean old vertices (overwrite)
    vertices = inp;
    calculate_outline();
}

template <class T, class W> void ALL<T,W>::set_vertices(const int n, const int dim, const T* inp)
{
    if (dim != dimension) throw ALL_Point_Dimension_Missmatch_Exception(
                                __FILE__,
                                __func__,
                                __LINE__,
                                "Dimension of ALL_Points in input vector do not match dimension of ALL object.");
    ALL_Point<double> tmp(dim);
    vertices = std::vector<ALL_Point<double>>(n,tmp);
    for (int i = 0; i < n; ++i)
    {
        for (int d = 0; d < dim; ++d)
        {
            vertices.at(i).set_coordinate(d,inp[ i * dim + d ]);
        }
    }
    calculate_outline();
}

template <class T, class W> void ALL<T,W>::calculate_outline()
{
    // calculation only possible if there are vertices defining the domain
    if (vertices.size() > 0)
    {
        // setup the outline with the first point
        for (int i = 0; i < dimension; ++i)
        {
            outline->at(0*dimension+i) = outline->at(1*dimension+i) = vertices.at(0).x(i);
        }
        // compare value of each outline point with all vertices to find the maximum
        // extend of the domain in each dimension
        for (int i = 1; i < vertices.size(); ++i)
        {
            ALL_Point<T> p = vertices.at(i);
            for( int j = 0; j < dimension; ++j)
            {
                outline->at(0*dimension+j) = std::min(outline->at(0*dimension+j),p.x(j));
                outline->at(1*dimension+j) = std::max(outline->at(1*dimension+j),p.x(j));
            }
        }
    }
}

template <class T, class W> void ALL<T,W>::set_work(W work)
{
    // clear work_array
    if (work_array) delete work_array;
    // set new value for work (single value for whole domain)
    work_array = new std::vector<W>(1);
    work_array->at(0) = work;
}

template <class T, class W> void ALL<T,W>::set_work(const std::vector<W>& work)
{
    if (work_array) delete work_array;
    work_array = new std::vector<W>(work);
}

template <class T, class W> void ALL<T,W>::set_communicator(MPI_Comm comm)
{
    communicator = comm;
}

template <class T, class W> void ALL<T,W>::get_work(W& result)
{
    result = work_array.at(0); 
}

template <class T, class W> void ALL<T,W>::get_work(std::vector<W>& result)
{
    result = work_array;
}

template <class T, class W> void ALL<T,W>::create_balancer(ALL_LB_t method)
{
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                balancer = new ALL_Tensor_LB<T,W>(dimension,work_array->at(0),gamma);
                break;
            case ALL_LB_t::STAGGERED:
                balancer = new ALL_Staggered_LB<T,W>(dimension,work_array->at(0),gamma);
                break;
            case ALL_LB_t::UNSTRUCTURED:
            	balancer = new ALL_Unstructured_LB<T,W>(dimension,work_array->at(0),gamma);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI                
            	balancer = new ALL_Voronoi_LB<T,W>(dimension,work_array->at(0),gamma);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
                balancer = new ALL_Histogram_LB<T,W>(dimension,work_array->at(0),gamma);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                __FILE__,
                                __func__,
                                __LINE__,
                                "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch (ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> void ALL<T,W>::setup(ALL_LB_t method)
{
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                if (balancer) delete (ALL_Tensor_LB<T,W>*)balancer;
                balancer = new ALL_Tensor_LB<T,W>(dimension,work_array->at(0),gamma);
                ((ALL_Tensor_LB<T,W>*)balancer)->set_vertices(outline->data());
                ((ALL_Tensor_LB<T,W>*)balancer)->setup(communicator);
                break;
            case ALL_LB_t::STAGGERED:
                if (balancer) delete (ALL_Staggered_LB<T,W>*)balancer;
                balancer = new ALL_Staggered_LB<T,W>(dimension,work_array->at(0),gamma);
                ((ALL_Staggered_LB<T,W>*)balancer)->set_vertices(outline->data());
                ((ALL_Staggered_LB<T,W>*)balancer)->setup(communicator);
                break;
            case ALL_LB_t::UNSTRUCTURED:
                if (balancer) delete (ALL_Unstructured_LB<T,W>*)balancer;
            	balancer = new ALL_Unstructured_LB<T,W>(dimension,work_array->at(0),gamma);
                ((ALL_Unstructured_LB<T,W>*)balancer)->set_vertices(vertices);
                ((ALL_Unstructured_LB<T,W>*)balancer)->setup(communicator);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
                if (balancer) delete (ALL_Voronoi_LB<T,W>*)balancer;
                balancer = new ALL_Voronoi_LB<T,W>(dimension,work_array->at(0),gamma);
                ((ALL_Voronoi_LB<T,W>*)balancer)->set_vertices(vertices);
                ((ALL_Voronoi_LB<T,W>*)balancer)->setup(communicator);
#endif                
                break;
            case ALL_LB_t::HISTOGRAM:
                if (balancer) delete (ALL_Histogram_LB<T,W>*)balancer;
                balancer = new ALL_Histogram_LB<T,W>(dimension,work_array,gamma);
                ((ALL_Histogram_LB<T,W>*)balancer)->set_vertices(outline->data());
                ((ALL_Histogram_LB<T,W>*)balancer)->setup(communicator);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch (ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> void ALL<T,W>::setup_balancer(ALL_LB_t method)
{
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                ((ALL_Tensor_LB<T,W>*)balancer)->setup(communicator);
                break;
            case ALL_LB_t::STAGGERED:
                ((ALL_Staggered_LB<T,W>*)balancer)->setup(communicator);
                break;
            case ALL_LB_t::UNSTRUCTURED:
                ((ALL_Unstructured_LB<T,W>*)balancer)->setup(communicator);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
                ((ALL_Voronoi_LB<T,W>*)balancer)->setup(communicator);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
                ((ALL_Histogram_LB<T,W>*)balancer)->setup(communicator);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch (ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> void ALL<T,W>::balance(ALL_LB_t method, bool internal)
{
    try
    {
        n_vertices = vertices.size();
        /*
    	n_vertices = 2;
        if (method == ALL_LB_t::UNSTRUCTURED)
        {
        	n_vertices = 8;
        }
        */
    	std::vector<T> result(n_vertices*dimension);
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                ((ALL_Tensor_LB<T,W>*)balancer)->set_vertices(outline->data());
                result_vertices = vertices;
                ((ALL_Tensor_LB<T,W>*)balancer)->balance();

                ((ALL_Tensor_LB<T,W>*)balancer)->get_shifted_vertices(result);

                for (int i = 0; i < result_vertices.size(); ++i)
                {
                    for (int d = 0; d < result_vertices.at(i).get_dimension(); ++d)
                    {
                        result_vertices.at(i).set_coordinate(
                                d,
                                result.at(i*result_vertices.at(i).get_dimension()+d)
                                );
                    }
                }
                
                break;
            case ALL_LB_t::STAGGERED:
                {
                    // estimation to improve speed of convergence
                    // changing vertices during estimation
                    std::vector<ALL_Point<T>> tmp_vertices = vertices;
                    // saved original vertices
                    std::vector<ALL_Point<T>> old_vertices = vertices;
                    // old temporary vertices
                    std::vector<ALL_Point<T>> old_tmp_vertices = vertices;
                    // temporary work
                    W tmp_work = work_array->at(0);
                    // old temporary work
                    W old_tmp_work = work_array->at(0);

                    W max_work;
                    W sum_work;
                    double d_max;
                    int n_ranks;

                    // compute work density on local domain
                    double V = 1.0;
                    for (int i = 0; i < dimension; ++i)
                        V *= ( outline->at(3+i) - outline->at(i) );
                    double rho = work_array->at(0) / V;

                    // collect maximum work in system
                    MPI_Allreduce(work_array->data(), &max_work, 1, mpi_data_type_W, MPI_MAX, communicator);
                    MPI_Allreduce(work_array->data(), &sum_work, 1, mpi_data_type_W, MPI_SUM, communicator);
                    MPI_Comm_size(communicator,&n_ranks);
                    d_max = (double)(max_work) * (double)n_ranks / (double)sum_work - 1.0;
                    

                    for (int i = 0; i < estimation_limit && d_max > 0.1 && !internal; ++i)
                    {
                        work_array->at(0) = tmp_work;
                        setup(method);
                        balance(method,true);
                        bool sane = true;
                        // check if the estimated boundaries are not too deformed
                        for (int j = 0; j < dimension; ++j)
                            sane = sane && (result_vertices.at(0).x(j) <= old_vertices.at(1).x(j));
                        MPI_Allreduce(MPI_IN_PLACE,&sane,1,MPI_CXX_BOOL,MPI_LAND,communicator);
                        if (sane)
                        {
                            old_tmp_vertices = tmp_vertices;
                            tmp_vertices = result_vertices;
                            V = 1.0;
                            for (int i = 0; i < dimension; ++i)
                                V *= ( tmp_vertices.at(1).x(i) - tmp_vertices.at(0).x(i) );
                            old_tmp_work = tmp_work;
                            tmp_work = rho * V;
                        }
                        else if (!sane || i == estimation_limit - 1)
                        {
                            vertices = old_tmp_vertices;
                            work_array->at(0) = old_tmp_work;
                            calculate_outline();
                            i = estimation_limit;
                        }
                    }

                    ((ALL_Staggered_LB<T,W>*)balancer)->set_vertices(outline->data());
                    result_vertices = vertices;
                    ((ALL_Staggered_LB<T,W>*)balancer)->balance();

                    ((ALL_Staggered_LB<T,W>*)balancer)->get_shifted_vertices(result);
                    for (int i = 0; i < result_vertices.size(); ++i)
                    {
                        for (int d = 0; d < result_vertices.at(i).get_dimension(); ++d)
                        {
                            result_vertices.at(i).set_coordinate(
                                    d,
                                    result.at(i*result_vertices.at(i).get_dimension()+d)
                                    );
                        }
                    }
                }
                break;
            case ALL_LB_t::UNSTRUCTURED:
            	((ALL_Unstructured_LB<T,W>*)balancer)->set_vertices(vertices);
                result_vertices = vertices;
                ((ALL_Unstructured_LB<T,W>*)balancer)->balance();
                ((ALL_Unstructured_LB<T,W>*)balancer)->get_shifted_vertices(result_vertices);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
            	((ALL_Voronoi_LB<T,W>*)balancer)->set_vertices(vertices);
            	((ALL_Voronoi_LB<T,W>*)balancer)->set_step(loadbalancing_step);
                result_vertices = vertices;
                ((ALL_Voronoi_LB<T,W>*)balancer)->balance();
                ((ALL_Voronoi_LB<T,W>*)balancer)->get_shifted_vertices(result_vertices);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
            	((ALL_Histogram_LB<T,W>*)balancer)->set_vertices(outline->data());
                result_vertices = vertices;
                ((ALL_Histogram_LB<T,W>*)balancer)->balance(loadbalancing_step);
                ((ALL_Histogram_LB<T,W>*)balancer)->get_shifted_vertices(result);
                for (int i = 0; i < result_vertices.size(); ++i)
                {
                    for (int d = 0; d < result_vertices.at(i).get_dimension(); ++d)
                    {
                        result_vertices.at(i).set_coordinate(
                                d,
                                result.at(i*result_vertices.at(i).get_dimension()+d)
                                );
                    }
                }
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
        loadbalancing_step++;
    }
    catch(ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> void ALL<T,W>::cleanup(ALL_LB_t method)
{
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                delete((ALL_Tensor_LB<T,W>*)balancer);
                break;
            case ALL_LB_t::STAGGERED:
                delete((ALL_Staggered_LB<T,W>*)balancer);
                break;
            case ALL_LB_t::UNSTRUCTURED:
                delete((ALL_Unstructured_LB<T,W>*)balancer);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
                delete((ALL_Voronoi_LB<T,W>*)balancer);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
                delete((ALL_Histogram_LB<T,W>*)balancer);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch(ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> int ALL<T,W>::get_n_neighbors(ALL_LB_t method)
{
    std::vector<int> neig;
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                ((ALL_Tensor_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::STAGGERED:
                ((ALL_Staggered_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::UNSTRUCTURED:
                ((ALL_Unstructured_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
                ((ALL_Voronoi_LB<T,W>*)balancer)->get_neighbors(neig);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
                ((ALL_Histogram_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch(ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
    return neig.size();
}

template <class T, class W> void ALL<T,W>::get_neighbor_vertices(ALL_LB_t method, std::vector<T>& nv)
{
    switch(method)
    {
        case ALL_LB_t::TENSOR:
            break;
        case ALL_LB_t::STAGGERED:
            break;
        case ALL_LB_t::UNSTRUCTURED:
            break;
        case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
            ((ALL_Voronoi_LB<T,W>*)balancer)->get_neighbor_vertices(nv);
#endif
            break;
        case ALL_LB_t::HISTOGRAM:
            break;
        default:
            break;
    }
}

template <class T, class W> void ALL<T,W>::get_neighbors(ALL_LB_t method, std::vector<int>& neig)
{
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                ((ALL_Tensor_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::STAGGERED:
                ((ALL_Staggered_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::UNSTRUCTURED:
                ((ALL_Unstructured_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
                ((ALL_Voronoi_LB<T,W>*)balancer)->get_neighbors(neig);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
                ((ALL_Histogram_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch(ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> void ALL<T,W>::get_neighbors(ALL_LB_t method, int** neig)
{
    try
    {
        switch(method)
        {
            case ALL_LB_t::TENSOR:
                ((ALL_Tensor_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::STAGGERED:
                ((ALL_Staggered_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::UNSTRUCTURED:
                ((ALL_Unstructured_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
                ((ALL_Voronoi_LB<T,W>*)balancer)->get_neighbors(neig);
#endif
                break;
            case ALL_LB_t::HISTOGRAM:
                ((ALL_Histogram_LB<T,W>*)balancer)->get_neighbors(neig);
                break;
            default:
                throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
                break;
        }
    }
    catch(ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

template <class T, class W> void ALL<T,W>::set_sys_size(ALL_LB_t method, std::vector<T>& s_size)
{
    switch(method)
    {
        case ALL_LB_t::TENSOR:
            break;
        case ALL_LB_t::STAGGERED:
            break;
        case ALL_LB_t::UNSTRUCTURED:
            break;
        case ALL_LB_t::VORONOI:
#ifdef ALL_VORONOI
            ((ALL_Voronoi_LB<T,W>*)balancer)->set_sys_size(s_size);
#endif
            break;
        case ALL_LB_t::HISTOGRAM:
            ((ALL_Histogram_LB<T,W>*)balancer)->set_sys_size(s_size);
            break;
        default:
            throw ALL_Invalid_Argument_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__,
                                        "Unknown type of loadbalancing passed.");
            break;
    }

}

template <class T, class W> void ALL<T,W>::set_method_data(ALL_LB_t method, void* data)
{
    switch(method)
    {
        case ALL_LB_t::TENSOR:
            break;
        case ALL_LB_t::STAGGERED:
            break;
        case ALL_LB_t::UNSTRUCTURED:
            break;
#ifdef ALL_VORONOI
        case ALL_LB_t::VORONOI:
            break;
#endif        
        case ALL_LB_t::HISTOGRAM:
            ((ALL_Histogram_LB<T,W>*)balancer)->set_data(data);
            break;
    }
}

#ifdef ALL_VTK_OUTPUT
template <class T, class W> void ALL<T,W>::print_vtk_outlines(int step)
{
    // define grid points, i.e. vertices of local domain
    auto points = vtkSmartPointer<vtkPoints>::New();
    // allocate space for eight points (describing the cuboid around the domain)
    points->Allocate(8 * dimension);

    int n_ranks;
    int local_rank;

    if (!vtk_init)
    {    
        controller = vtkMPIController::New();
        controller->Initialize();
        controller->SetGlobalController(controller);
        vtk_init = true;
    }

    MPI_Comm_rank(communicator,&local_rank);
    MPI_Comm_size(communicator,&n_ranks);

    std::vector<ALL_Point<W>> tmp_outline(2);
    tmp_outline.at(0) = ALL_Point<W>(3,outline->data());
    tmp_outline.at(1) = ALL_Point<W>(3,outline->data()+3);

    for (auto z = 0; z <= 1; ++z)
        for (auto y = 0; y <= 1; ++y)
            for (auto x = 0; x <= 1; ++x)
            {
                points->InsertPoint(x + 2 * y + 4 * z,
                                    tmp_outline.at(x).x(0),
                                    tmp_outline.at(y).x(1),
                                    tmp_outline.at(z).x(2));
            }

    auto hexa = vtkSmartPointer<vtkVoxel>::New();
    for (int i = 0; i < 8; ++i)
        hexa->GetPointIds()->SetId(i,i);

    auto cellArray = vtkSmartPointer<vtkCellArray>::New();
    cellArray->InsertNextCell(hexa);

    // define work array (length = 1)
    auto work = vtkSmartPointer<vtkFloatArray>::New();
    work->SetNumberOfComponents(1);
    work->SetNumberOfTuples(1);
    work->SetName("work");
    work->SetValue(0, work_array->at(0));

    // determine extent of system
    W global_extent[6];
    W local_min[3];
    W local_max[3];
    W global_min[3];
    W global_max[3];

    for(int i = 0; i < 3; ++i)
    {
        local_min[i] = outline->at(i);
        local_max[i] = outline->at(3+i);
    } 

    MPI_Allreduce(&local_min,&global_min,3,mpi_data_type_W,MPI_MIN,communicator);
    MPI_Allreduce(&local_max,&global_max,3,mpi_data_type_W,MPI_MAX,communicator);

    for(int i = 0; i < 3; ++i)
    {
        global_extent[2*i]   = global_min[i];
        global_extent[2*i+1] = global_max[i];
    }

    // create a structured grid and assign data to it
    auto unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    unstructuredGrid->SetPoints(points);
    unstructuredGrid->SetCells(VTK_VOXEL, cellArray);
    unstructuredGrid->GetCellData()->AddArray(work);

    

    std::ostringstream ss_local;
    ss_local << "vtk_outline/ALL_vtk_outline_" << std::setw(7) << std::setfill('0') 
             << step << "_" << local_rank << ".vtu";

    auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetInputData(unstructuredGrid);
    writer->SetFileName(ss_local.str().c_str());
    writer->SetDataModeToAscii();
    //writer->SetDataModeToBinary();
    writer->Write();

    //if (local_rank == 0)
    //{
        std::ostringstream ss_para;
        ss_para << "vtk_outline/ALL_vtk_outline_" << std::setw(7) << std::setfill('0') 
                << step << ".pvtu";
        // create the parallel writer
        auto parallel_writer = vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
        parallel_writer->SetFileName(ss_para.str().c_str());
        parallel_writer->SetNumberOfPieces(n_ranks);
        parallel_writer->SetStartPiece(local_rank);
        parallel_writer->SetEndPiece(local_rank);
        parallel_writer->SetInputData(unstructuredGrid);
        parallel_writer->SetDataModeToAscii();
        //parallel_writer->SetDataModeToBinary();
        parallel_writer->Write();
    //}
}

template <class T, class W> void ALL<T,W>::print_vtk_vertices(int step)
{
    try
    {
        int n_ranks;
        int local_rank;

        MPI_Comm_rank(communicator,&local_rank);
        MPI_Comm_size(communicator,&n_ranks);

        // local vertices
        // (vertices + work)
        T local_vertices[n_vertices * dimension + 1];

        for (int v = 0; v < n_vertices; ++v)
        {
            for (int d = 0; d < dimension; ++d)
            {
                local_vertices[v * dimension + d] = vertices.at(v).x(d);
            }
        }
        local_vertices[n_vertices * dimension] = (T)work_array->at(0);

        T* global_vertices;
        if (local_rank == 0)
        {
            global_vertices = new T[n_ranks * (n_vertices * dimension + 1)];
        }

        // collect all works and vertices on a single process
        MPI_Gather(local_vertices, n_vertices * dimension + 1, mpi_data_type_T,
                   global_vertices,n_vertices * dimension + 1, mpi_data_type_T,
                   0, communicator
                  );

        if (local_rank == 0)
        {
            auto vtkpoints = vtkSmartPointer<vtkPoints>::New();
            auto unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    
            // enter vertices into unstructured grid
            for (int i = 0; i < n_ranks; ++i)
            {
                for (int v = 0; v < n_vertices; ++v)
                {
                    vtkpoints->InsertNextPoint(
                        global_vertices[ i * (n_vertices * dimension + 1) + v * dimension + 0 ],
                        global_vertices[ i * (n_vertices * dimension + 1) + v * dimension + 1 ],
                        global_vertices[ i * (n_vertices * dimension + 1) + v * dimension + 2 ]
                            );
                }
            }
            unstructuredGrid->SetPoints(vtkpoints);

            // data arrays for work and cell id
            auto work = vtkSmartPointer<vtkFloatArray>::New();
            auto cell = vtkSmartPointer<vtkFloatArray>::New();
            work->SetNumberOfComponents(1);
            work->SetNumberOfTuples(n_ranks);
            work->SetName("work");
            cell->SetNumberOfComponents(1);
            cell->SetNumberOfTuples(n_ranks);
            cell->SetName("cell id");

            for (int n = 0; n < n_ranks; ++n)
            {
                // define grid points, i.e. vertices of local domain
                vtkIdType pointIds[8] = { 8 * n + 0, 8 * n + 1, 8 * n + 2, 8 * n + 3, 
                                          8 * n + 4, 8 * n + 5, 8 * n + 6, 8 * n + 7 };

                auto faces = vtkSmartPointer<vtkCellArray>::New();
                // setup faces of polyhedron
                vtkIdType f0[3] = {8 * n + 0, 8 * n + 2, 8 * n + 1};
                vtkIdType f1[3] = {8 * n + 1, 8 * n + 2, 8 * n + 3};

                vtkIdType f2[3] = {8 * n + 0, 8 * n + 4, 8 * n + 2};
                vtkIdType f3[3] = {8 * n + 2, 8 * n + 4, 8 * n + 6};

                vtkIdType f4[3] = {8 * n + 2, 8 * n + 6, 8 * n + 3};
                vtkIdType f5[3] = {8 * n + 3, 8 * n + 6, 8 * n + 7};

                vtkIdType f6[3] = {8 * n + 1, 8 * n + 5, 8 * n + 3};
                vtkIdType f7[3] = {8 * n + 3, 8 * n + 5, 8 * n + 7};

                vtkIdType f8[3] = {8 * n + 0, 8 * n + 4, 8 * n + 1};
                vtkIdType f9[3] = {8 * n + 1, 8 * n + 4, 8 * n + 5};

                vtkIdType fa[3] = {8 * n + 4, 8 * n + 6, 8 * n + 5};
                vtkIdType fb[3] = {8 * n + 5, 8 * n + 6, 8 * n + 7};

                faces->InsertNextCell(3,f0);
                faces->InsertNextCell(3,f1);
                faces->InsertNextCell(3,f2);
                faces->InsertNextCell(3,f3);
                faces->InsertNextCell(3,f4);
                faces->InsertNextCell(3,f5);
                faces->InsertNextCell(3,f6);
                faces->InsertNextCell(3,f7);
                faces->InsertNextCell(3,f8);
                faces->InsertNextCell(3,f9);
                faces->InsertNextCell(3,fa);
                faces->InsertNextCell(3,fb);

                unstructuredGrid->InsertNextCell(VTK_POLYHEDRON, 8, pointIds,
                                                 12, faces->GetPointer());
                work->SetValue(n,global_vertices[ n * (n_vertices * dimension + 1) + 
                                                  8 * dimension ]);
                cell->SetValue(n,(T)n);
            }
            unstructuredGrid->GetCellData()->AddArray(work);
            unstructuredGrid->GetCellData()->AddArray(cell);

            std::ostringstream filename;
            filename << "vtk_vertices/ALL_vtk_vertices_" << 
                        std::setw(7) << std::setfill('0') << step << ".vtu";
            auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
            writer->SetInputData(unstructuredGrid);
            writer->SetFileName(filename.str().c_str());
            writer->SetDataModeToAscii();
            //writer->SetDataModeToBinary();
            writer->Write();

            delete[] global_vertices;
        }


        /*
        
        // define grid points, i.e. vertices of local domain
        vtkIdType pointIds[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

        auto points = vtkSmartPointer<vtkPoints>::New();

        int n_ranks;
        int local_rank;

        if (!vtk_init)
        {    
            controller = vtkMPIController::New();
            controller->Initialize();
            controller->SetGlobalController(controller_vertices);
            vtk_init = true;
        }

        MPI_Comm_rank(communicator,&local_rank);
        MPI_Comm_size(communicator,&n_ranks);

        // setup vertices of polyhedron
        if (n_vertices == 2)
        {
            for (auto z = 0; z <= 1; ++z)
                for (auto y = 0; y <= 1; ++y)
                    for (auto x = 0; x <= 1; ++x)
                    {
                        points->InsertPoint(x + 2 * y + 4 * z,
                                            vertices.at(x).x(0),
                                            vertices.at(y).x(1),
                                            vertices.at(z).x(2));
                    }
        }
        else if (n_vertices == 8)
        {
            for (int i = 0; i < 8; ++i)
            {
                points->InsertNextPoint(
                          vertices.at(i).x(0),
                          vertices.at(i).x(1),
                          vertices.at(i).x(2)
                                       );
            }
        }
        else
        {
            throw ALL_Invalid_Argument_Exception(   "ALL.hpp",
                                                    "ALL<T,W>::print_vtk_vertices(int)",
                                                    957,
                                                    "Unsupported number of vertices.");
        }

        // setup faces of polyhedron
        auto faces = vtkSmartPointer<vtkCellArray>::New();
        vtkIdType f0[3] = {0, 2, 1};
        vtkIdType f1[3] = {1, 2, 3};

        vtkIdType f2[3] = {0, 4, 2};
        vtkIdType f3[3] = {2, 4, 6};

        vtkIdType f4[3] = {2, 6, 3};
        vtkIdType f5[3] = {3, 6, 7};

        vtkIdType f6[3] = {1, 5, 3};
        vtkIdType f7[3] = {3, 5, 7};

        vtkIdType f8[3] = {0, 4, 1};
        vtkIdType f9[3] = {1, 4, 5};

        vtkIdType fa[3] = {4, 6, 5};
        vtkIdType fb[3] = {5, 6, 7};

        faces->InsertNextCell(3,f0);
        faces->InsertNextCell(3,f1);
        faces->InsertNextCell(3,f2);
        faces->InsertNextCell(3,f3);
        faces->InsertNextCell(3,f4);
        faces->InsertNextCell(3,f5);
        faces->InsertNextCell(3,f6);
        faces->InsertNextCell(3,f7);
        faces->InsertNextCell(3,f8);
        faces->InsertNextCell(3,f9);
        faces->InsertNextCell(3,fa);
        faces->InsertNextCell(3,fb);

        // define work array (length = 1)
        auto work = vtkSmartPointer<vtkFloatArray>::New();
        work->SetNumberOfComponents(1);
        work->SetNumberOfTuples(1);
        work->SetName("work");
        work->SetValue(0, work_array->at(0));

        // determine extent of system
        W global_extent[6];
        W local_min[3];
        W local_max[3];
        W global_min[3];
        W global_max[3];

        for(int i = 0; i < 3; ++i)
        {
            local_min[i] = outline->at(i);
            local_max[i] = outline->at(3+i);
        } 

        MPI_Allreduce(&local_min,&global_min,3,mpi_data_type_W,MPI_MIN,communicator);
        MPI_Allreduce(&local_max,&global_max,3,mpi_data_type_W,MPI_MAX,communicator);

        for(int i = 0; i < 3; ++i)
        {
            global_extent[2*i]   = global_min[i];
            global_extent[2*i+1] = global_max[i];
        }

        // create a structured grid and assign data to it
        auto unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        unstructuredGrid->SetPoints(points);
        unstructuredGrid->InsertNextCell(VTK_POLYHEDRON, 8, pointIds,
                          12, faces->GetPointer());
        unstructuredGrid->GetCellData()->AddArray(work);

        

        std::ostringstream ss_local;
        ss_local << "vtk_vertices/ALL_vtk_vertices_" << std::setw(7) << std::setfill('0') << step << "_" << local_rank << ".vtu";

        auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetInputData(unstructuredGrid);
        writer->SetFileName(ss_local.str().c_str());
        writer->SetDataModeToAscii();
        //writer->SetDataModeToBinary();
        writer->Write();

        //if (local_rank == 0)
        //{
            std::ostringstream ss_para;
            ss_para << "vtk_vertices/ALL_vtk_vertices_" << std::setw(7) << std::setfill('0') << step << ".pvtu";
            // create the parallel writer
            auto parallel_writer = vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
            parallel_writer->SetFileName(ss_para.str().c_str());
            parallel_writer->SetNumberOfPieces(n_ranks);
            parallel_writer->SetStartPiece(local_rank);
            parallel_writer->SetEndPiece(local_rank);
            parallel_writer->SetInputData(unstructuredGrid);
            parallel_writer->SetDataModeToAscii();
            //parallel_writer->SetDataModeToBinary();
            parallel_writer->Update();
            parallel_writer->Write();
        //}

        */            
    }
    catch (ALL_Custom_Exception e)
    {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}
#endif

#endif
