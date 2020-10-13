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

#ifndef ALL_TENSOR_HEADER_INCLUDED
#define ALL_TENSOR_HEADER_INCLUDED

#include <mpi.h>
#include <exception>
#include "ALL_CustomExceptions.hpp"

// T: data type of vertices used in the balancer
// W: data type of the work used in the balancer
template <class T, class W> class ALL_Tensor_LB
{
    public:
        ALL_Tensor_LB() {}
        ALL_Tensor_LB(int d, W w, T g) : dimension(d),
                                         work(w), 
                                         gamma(g)
        {
            // need the lower and upper bounds
            // of the domain for the tensor-based
            // load-balancing scheme
            vertices = new T[2*dimension];
            shifted_vertices = new T[2*dimension];
            // global dimension of the system in each coordinate
            global_dims = new int[dimension];
            // coordinates of the local domain in the system
            local_coords = new int[dimension];
            // periodicity of the MPI communicator / system
            periodicity = new int[dimension];

            // array of MPI communicators for each direction (to collect work
            // on each plane)
            communicators = new MPI_Comm[dimension];
            n_neighbors = new int[2*dimension];
        }

        ~ALL_Tensor_LB(); 

        void set_vertices(T*);

        // setup communicators for the exchange of work in each domain
        void setup(MPI_Comm);

        // do a load-balancing step
        virtual void balance();

        // getter for variables (passed by reference to avoid
        // direct access to private members of the object)

        // getter for base vertices
        void get_vertices(T*);
        // getter for shifted vertices
        void get_shifted_vertices(T*);
        // getter for shifted vertices
        void get_shifted_vertices(std::vector<T>&);

        // neighbors
        // provide list of neighbors
        virtual void get_neighbors(std::vector<int>&);
        // provide list of neighbors in each direction
        virtual void get_neighbors(int**);


    private:
        // dimension of the system
        int dimension;

        // shift cooeficient
        T gamma;

        // work per domain
        W work;

        // vertices
        T* vertices; 

        // vertices after load-balancing shift
        T* shifted_vertices; 

        // MPI values
        // global communicator
        MPI_Comm global_comm;
        // rank of the local domain in the communicator
        int local_rank;
        // global dimension of the system in each coordinate
        int* global_dims;
        // coordinates of the local domain in the system
        int* local_coords;
        // periodicity of the MPI communicator / system
        int* periodicity;
        
        // type for MPI communication
        MPI_Datatype mpi_data_type_T;
        MPI_Datatype mpi_data_type_W;

        // array of MPI communicators for each direction (to collect work
        // on each plane)
        MPI_Comm* communicators;

        // list of neighbors
        std::vector<int> neighbors;
        int* n_neighbors;
};

template <class T, class W> ALL_Tensor_LB<T,W>::~ALL_Tensor_LB()
{
    if (vertices) delete vertices;
    if (shifted_vertices) delete shifted_vertices;
    if (global_dims) delete global_dims;
    if (local_coords) delete local_coords;
    if (periodicity) delete periodicity;
    if (communicators) delete communicators;
    if (n_neighbors) delete n_neighbors;
}

template <class T, class W> void ALL_Tensor_LB<T,W>::get_vertices(T* result)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        result[i] = vertices[i];
    }
}

template <class T, class W> void ALL_Tensor_LB<T,W>::get_shifted_vertices(T* result)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        result[i] = shifted_vertices[i];
    }
}

template <class T, class W> void ALL_Tensor_LB<T,W>::get_shifted_vertices(std::vector<T>& result)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        result.at(i) = shifted_vertices[i];
    }
}

// set the actual vertices (unsafe due to array copy)
template <class T, class W> void ALL_Tensor_LB<T,W>::set_vertices(T* v)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        vertices[i] = v[i];
    }
}

// setup routine for the tensor-based load-balancing scheme
// requires: 
//              global_comm (int): cartesian MPI communicator, from
//                                 which separate sub communicators
//                                 are derived in order to represent
//                                 each plane of domains in the system
template <class T, class W> void ALL_Tensor_LB<T,W>::setup(MPI_Comm comm)
{
    int status;

    // check if Communicator is cartesian
    MPI_Topo_test(comm, &status);
    if (status != MPI_CART)
    {
        throw ALL_Invalid_Comm_Type_Exception(
                __FILE__,
                __func__,
                __LINE__,
                "Cartesian MPI communicator required, passed communicator not cartesian");
    }
    
    // allocate arrays to correct sizes
    global_dims = new int[dimension];
    local_coords = new int[dimension];
    periodicity = new int[dimension];

    // store MPI communicator to local variable
    global_comm = comm;

    // get the local coordinates, periodicity and global size from the MPI communicator
    MPI_Cart_get(global_comm, dimension, global_dims, periodicity, local_coords);

    // get the local rank from the MPI communicator
    MPI_Cart_rank(global_comm, local_coords, &local_rank);

    // create sub-communicators 
    for (int i = 0; i < dimension; ++i)
    {
        MPI_Comm_split(global_comm,local_coords[i],0,&communicators[i]);
    }

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

    // calculate neighbors
    int rank_left, rank_right;

    neighbors.clear();
    for (int i = 0; i < dimension; ++i)
    {
        MPI_Cart_shift(global_comm,i,1,&rank_left,&rank_right);
        neighbors.push_back(rank_left);
        neighbors.push_back(rank_right);
        n_neighbors[2*i] = 1;
        n_neighbors[2*i+1] = 1;
    }
    
}

template<class T, class W> void ALL_Tensor_LB<T,W>::balance()
{
    // loop over all available dimensions
    for (int i = 0; i < dimension; ++i)
    {
        W work_local_plane;
        // collect work from all processes in the same plane
        MPI_Allreduce(&work,&work_local_plane,1,mpi_data_type_W,MPI_SUM,communicators[i]);
        
        // correct right border:

        W remote_work;
        T size;
        T remote_size;
        // determine neighbors
        int rank_left, rank_right;

        MPI_Cart_shift(global_comm,i,1,&rank_left,&rank_right);

        // collect work from right neighbor plane
        MPI_Request sreq, rreq;
        MPI_Status sstat, rstat;

        MPI_Irecv(&remote_work,1,mpi_data_type_W,rank_right,0,global_comm,&rreq);
        MPI_Isend(&work_local_plane,1,mpi_data_type_W,rank_left,0,global_comm,&sreq);
        MPI_Wait(&sreq,&sstat);
        MPI_Wait(&rreq,&rstat);

        // collect size in dimension from right neighbor plane

        size = vertices[dimension+i] - vertices[i];
        
        MPI_Irecv(&remote_size,1,mpi_data_type_T,rank_right,0,global_comm,&rreq);
        MPI_Isend(&size,1,mpi_data_type_T,rank_left,0,global_comm,&sreq);
        MPI_Wait(&sreq,&sstat);
        MPI_Wait(&rreq,&rstat);

        // calculate shift of borders:
        // s = 0.5 * gamma * (W_r - W_l) / (W_r + W_l) * (d_r + d_l)
        // *_r = * of neighbor (remote)
        // *_l = * of local domain
        T shift = (T)0;
        if (rank_right != MPI_PROC_NULL && local_coords[i] != global_dims[i]-1 && !( remote_work == (T)0 && work_local_plane == (T)0) )
            shift = 1.0 / gamma * 0.5 * (remote_work - work_local_plane) / (remote_work + work_local_plane) * (size + remote_size);

        if (shift < 0.0)
        {
            if (abs(shift) > 0.45 * size) shift = -0.45 * size;
        }
        else
        {
            if (abs(shift) > 0.45 * remote_size) shift = 0.45 * remote_size;
        }


        // send shift to right neighbors
        T remote_shift = (T)0;
        
        MPI_Irecv(&remote_shift,1,mpi_data_type_T,rank_left,0,global_comm,&rreq);
        MPI_Isend(&shift,1,mpi_data_type_T,rank_right,0,global_comm,&sreq);
        MPI_Wait(&sreq,&sstat);
        MPI_Wait(&rreq,&rstat);

        // for now: test case for simple program

        // if a left neighbor exists: shift left border 
        if (rank_left != MPI_PROC_NULL && local_coords[i] != 0)
            shifted_vertices[i] = vertices[i] + remote_shift;
        else
            shifted_vertices[i] = vertices[i];

        // if a right neighbor exists: shift right border
        if (rank_right != MPI_PROC_NULL && local_coords[i] != global_dims[i]-1)
            shifted_vertices[dimension+i] = vertices[dimension+i] + shift;
        else
            shifted_vertices[dimension+i] = vertices[dimension+i];

    }
}

// provide list of neighbors
template<class T, class W> void ALL_Tensor_LB<T,W>::get_neighbors(std::vector<int>& ret)
{
    ret = neighbors;
}
// provide list of neighbors in each direction
template<class T, class W> void ALL_Tensor_LB<T,W>::get_neighbors(int** ret)
{
    *ret = n_neighbors;
}
#endif
