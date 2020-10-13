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

#ifndef ALL_STAGGERED_HEADER_INCLUDED
#define ALL_STAGGERED_HEADER_INCLUDED

#include <mpi.h>
#include <exception>
#include <vector>
#include "ALL_CustomExceptions.hpp"

// T: data type of vertices used in the balancer
// W: data type of the work used in the balancer
template <class T, class W> class ALL_Staggered_LB
{
    public:
        ALL_Staggered_LB() {}
        ALL_Staggered_LB(int d, W w, T g) : dimension(d),
                                         work(w), 
                                         gamma(g)
        {
            // need the lower and upper bounds
            // of the domain for the tensor-based
            // load-balancing scheme
            vertices = new std::vector<T>(6);
            shifted_vertices = new std::vector<T>(6);
            // global dimension of the system in each coordinate
            global_dims = new int[3];
            // coordinates of the local domain in the system
            local_coords = new int[3];
            // periodicity of the MPI communicator / system
            periodicity = new int[3];

            // array of MPI communicators for each direction (to collect work
            // on each plane)
            communicators = new MPI_Comm[3];
            n_neighbors = new std::vector<int>(6);
        }

        ~ALL_Staggered_LB(); 

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
        std::vector<T>* vertices; 

        // vertices after load-balancing shift
        std::vector<T>* shifted_vertices; 

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
        std::vector<int>* n_neighbors;

        // find neighbors (more sophisticated than for tensor-product
        // variant of LB)
        void find_neighbors();
};

template <class T, class W> ALL_Staggered_LB<T,W>::~ALL_Staggered_LB()
{
    if (vertices) delete vertices;
    if (shifted_vertices) delete shifted_vertices;
    if (global_dims) delete global_dims;
    if (local_coords) delete local_coords;
    if (periodicity) delete periodicity;
    if (communicators)
    {
        for (int i = 1; i < 3; ++i)
            MPI_Comm_free(communicators + i);
        delete communicators;
    }
    if (n_neighbors) delete n_neighbors;
}

template <class T, class W> void ALL_Staggered_LB<T,W>::get_vertices(T* result)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        result[i] = vertices->at(i);
    }
}

template <class T, class W> void ALL_Staggered_LB<T,W>::get_shifted_vertices(T* result)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        result[i] = shifted_vertices->at(i);
    }
}

template <class T, class W> void ALL_Staggered_LB<T,W>::get_shifted_vertices(std::vector<T>& result)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        result.at(i) = shifted_vertices->at(i);
    }
}

// set the actual vertices (unsafe due to array copy)
template <class T, class W> void ALL_Staggered_LB<T,W>::set_vertices(T* v)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        vertices->at(i) = v[i];
    }
}

// setup routine for the tensor-based load-balancing scheme
// requires: 
//              global_comm (int): cartesian MPI communicator, from
//                                 which separate sub communicators
//                                 are derived in order to represent
//                                 each plane of domains in the system
template <class T, class W> void ALL_Staggered_LB<T,W>::setup(MPI_Comm comm)
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
                "Cartesian MPI communicator required, passed communicator is not cartesian");
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

    // z-plane
    MPI_Comm_split(global_comm,local_coords[2],local_coords[0]+local_coords[1]*global_dims[0],&communicators[2]);

    // y-column
    MPI_Comm_split(global_comm,local_coords[2] * global_dims[1]
                              +local_coords[1] ,local_coords[0],
                              &communicators[1]);

    // only cell itself
    communicators[0] = MPI_COMM_SELF;

    // determine correct MPI data type for template T
    if (std::is_same<T,double>::value) mpi_data_type_T = MPI_DOUBLE;
    else if (std::is_same<T,float>::value) mpi_data_type_T = MPI_FLOAT;
    else if (std::is_same<T,int>::value) mpi_data_type_T = MPI_INT;
    else if (std::is_same<T,long>::value) mpi_data_type_T = MPI_LONG;
    else
    {
        throw ALL_Invalid_Comm_Type_Exception(
                __FILE__,
                __func__,
                __LINE__,
                "Invalid data type for boundaries given (T)");
    }

    // determine correct MPI data type for template W 
    if (std::is_same<W,double>::value) mpi_data_type_W = MPI_DOUBLE;
    else if (std::is_same<W,float>::value) mpi_data_type_W = MPI_FLOAT;
    else if (std::is_same<W,int>::value) mpi_data_type_W = MPI_INT;
    else if (std::is_same<W,long>::value) mpi_data_type_W = MPI_LONG;
    else
    {
        throw ALL_Invalid_Comm_Type_Exception(
                __FILE__,
                __func__,
                __LINE__,
                "Invalid data type for work given (W)");
    }

    // calculate neighbors
    int rank_left, rank_right;

    neighbors.clear();
    for (int i = 0; i < dimension; ++i)
    {
        MPI_Cart_shift(global_comm,i,1,&rank_left,&rank_right);
        neighbors.push_back(rank_left);
        neighbors.push_back(rank_right);
    }
    
}

template<class T, class W> void ALL_Staggered_LB<T,W>::balance()
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

        size = vertices->at(dimension+i) - vertices->at(i);
        
        MPI_Irecv(&remote_size,1,mpi_data_type_T,rank_right,0,global_comm,&rreq);
        MPI_Isend(&size,1,mpi_data_type_T,rank_left,0,global_comm,&sreq);
        MPI_Wait(&sreq,&sstat);
        MPI_Wait(&rreq,&rstat);

        // automatic selection of gamma to guarantee stability of method (choice of user gamma ignored)
        gamma = std::max(4.1, 1.1 * (1.0 + std::max(size,remote_size) / std::min(size,remote_size)));

        // calculate shift of borders:
        // s = 0.5 * gamma * (W_r - W_l) / (W_r + W_l) * (d_r + d_l)
        // *_r = * of neighbor (remote)
        // *_l = * of local domain
        T shift = (T)0;
        if (rank_right != MPI_PROC_NULL && local_coords[i] != global_dims[i]-1 && !( remote_work == (T)0 && work_local_plane == (T)0) )
            shift = 1.0 / gamma * 0.5 * (remote_work - work_local_plane) / (remote_work + work_local_plane) * (size + remote_size);

        // reimplemented check if move is too large
        // in theory should be unneeded due to automatic choice of gamma
        T maxmove = 0.4 * std::min(size,remote_size);

        if (std::fabs(shift) > maxmove)
        {
            shift = (shift < 0.0)?-maxmove:maxmove;
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
            shifted_vertices->at(i) = vertices->at(i) + remote_shift;
        else
            shifted_vertices->at(i) = vertices->at(i);

        // if a right neighbor exists: shift right border
        if (rank_right != MPI_PROC_NULL && local_coords[i] != global_dims[i]-1)
            shifted_vertices->at(dimension+i) = vertices->at(dimension+i) + shift;
        else
            shifted_vertices->at(dimension+i) = vertices->at(dimension+i);

        // check if vertices are crossed and throw exception if thing went wrong
        if (shifted_vertices->at(dimension+i) < shifted_vertices->at(i))
        {
            std::cout << "ERROR on process: " << local_rank << " "
                                              << vertices->at(i) << " "
                                              << vertices->at(dimension+i) << " "
                                              << shifted_vertices->at(i) << " "
                                              << shifted_vertices->at(dimension+i) << " "
                                              << remote_shift << " "
                                              << shift << " "
                                              << remote_size << " "
                                              << size << " "
                                              << std::fabs(maxmove) << " "
                                              << maxmove << " "
                                              << std::endl;
            throw ALL_Internal_Error_Exception(
                    __FILE__,
                    __func__,
                    __LINE__,
                    "Lower border of process larger than upper border of process!");
        }

        find_neighbors();
    }
}

template<class T, class W> void ALL_Staggered_LB<T,W>::find_neighbors()
{
    neighbors.clear();
    // collect work from right neighbor plane
    MPI_Request sreq, rreq;
    MPI_Status sstat, rstat;
    // array to store neighbor vertices in Y/Z direction (reused)
    T* vertices_loc = new T[4];
    T* vertices_rem = new T[8*global_dims[0]*global_dims[1]];

    int rem_rank;
    int rem_coords[3];

    // determine neighbors
    int rank_left, rank_right;

    // offset to get the correct rank
    int rank_offset;
    int offset_coords[3];

    // X-neighbors are static
    n_neighbors->at(0) = n_neighbors->at(1) = 1;

    // find X-neighbors
    MPI_Cart_shift(global_comm,0,1,&rank_left,&rank_right);

    // store X-neighbors 
    neighbors.push_back(rank_left);
    neighbors.push_back(rank_right);

    // find Y-neighbors to get border information from
    MPI_Cart_shift(global_comm,1,1,&rank_left,&rank_right);

    // collect border information from local column
    vertices_loc[0] = shifted_vertices->at(0);
    vertices_loc[1] = shifted_vertices->at(dimension);
    MPI_Allgather(vertices_loc,2,mpi_data_type_T,vertices_rem+2*global_dims[0],2,mpi_data_type_T,communicators[1]);

    // exchange local column information with upper neighbor in Y direction (cart grid)
    MPI_Irecv(vertices_rem                 ,2*global_dims[0],mpi_data_type_T,rank_left, 0,global_comm,&rreq);
    MPI_Isend(vertices_rem+2*global_dims[0],2*global_dims[0],mpi_data_type_T,rank_right,0,global_comm,&sreq);

    // determine the offset in ranks
    offset_coords[0] = 0;
    offset_coords[1] = local_coords[1] - 1;
    offset_coords[2] = local_coords[2];

    rem_coords[1] = offset_coords[1];
    rem_coords[2] = offset_coords[2];

    MPI_Cart_rank(global_comm,offset_coords,&rank_offset);

    // wait for communication
    MPI_Wait(&sreq,&sstat);
    MPI_Wait(&rreq,&rstat);

    // iterate about neighbor borders to determine the neighborship relation
    n_neighbors->at(2) = 0;
    for (int x = 0; x < global_dims[0]; ++x)
    {
        if ( 
                ( vertices_rem[2*x] <= vertices_loc[0] && vertices_loc[0] <  vertices_rem[2*x+1] ) ||
                ( vertices_rem[2*x] <  vertices_loc[1] && vertices_loc[1] <= vertices_rem[2*x+1] ) ||
                ( vertices_rem[2*x] >= vertices_loc[0] && vertices_loc[0] < vertices_rem[2*x+1] &&
                  vertices_loc[1] >= vertices_rem[2*x+1] )
           )
        {
            n_neighbors->at(2)++;
            rem_coords[0] = x;
            MPI_Cart_rank(global_comm,rem_coords,&rem_rank);
            neighbors.push_back(rem_rank);
        }
    }

    // barrier to ensure every process concluded the calculations before overwriting remote borders!
    MPI_Barrier(global_comm);

    // exchange local column information with lower neighbor in Y direction (cart grid)
    MPI_Irecv(vertices_rem                 ,2*global_dims[0],mpi_data_type_T,rank_right, 0,global_comm,&rreq);
    MPI_Isend(vertices_rem+2*global_dims[0],2*global_dims[0],mpi_data_type_T,rank_left,  0,global_comm,&sreq);

    // determine the offset in ranks
    offset_coords[0] = 0;
    offset_coords[1] = local_coords[1] + 1;
    offset_coords[2] = local_coords[2];

    rem_coords[1] = offset_coords[1];
    rem_coords[2] = offset_coords[2];

    MPI_Cart_rank(global_comm,offset_coords,&rank_offset);

    // wait for communication
    MPI_Wait(&sreq,&sstat);
    MPI_Wait(&rreq,&rstat);

    // iterate about neighbor borders to determine the neighborship relation
    n_neighbors->at(3) = 0;
    for (int x = 0; x < global_dims[0]; ++x)
    {
        if ( 
                ( vertices_rem[2*x] <= vertices_loc[0] && vertices_loc[0] <  vertices_rem[2*x+1] ) ||
                ( vertices_rem[2*x] <  vertices_loc[1] && vertices_loc[1] <= vertices_rem[2*x+1] ) ||
                ( vertices_rem[2*x] >= vertices_loc[0] && vertices_loc[0] < vertices_rem[2*x+1] &&
                  vertices_loc[1] >= vertices_rem[2*x+1] )
           )
        {
            n_neighbors->at(3)++;
            rem_coords[0] = x;
            MPI_Cart_rank(global_comm,rem_coords,&rem_rank);
            neighbors.push_back(rem_rank);
        }
    }

    // barrier to ensure every process concluded the calculations before overwriting remote borders!
    MPI_Barrier(global_comm);

    // find Z-neighbors to get border information from
    MPI_Cart_shift(global_comm,2,1,&rank_left,&rank_right);

    // collect border information from local column
    vertices_loc[0] = shifted_vertices->at(0);
    vertices_loc[1] = shifted_vertices->at(dimension);
    vertices_loc[2] = shifted_vertices->at(1);
    vertices_loc[3] = shifted_vertices->at(1+dimension);

    MPI_Barrier(global_comm);

    MPI_Allgather(vertices_loc,4,mpi_data_type_T,vertices_rem+4*global_dims[0]*global_dims[1],4,mpi_data_type_T,communicators[2]);

    // exchange local column information with upper neighbor in Z direction (cart grid)
    MPI_Irecv(vertices_rem                                ,4*global_dims[0]*global_dims[1],mpi_data_type_T,rank_left, 0,global_comm,&rreq);
    MPI_Isend(vertices_rem+4*global_dims[0]*global_dims[1],4*global_dims[0]*global_dims[1],mpi_data_type_T,rank_right,0,global_comm,&sreq);

    // determine the offset in ranks
    offset_coords[0] = 0;
    offset_coords[1] = 0;
    offset_coords[2] = local_coords[2]-1;

    rem_coords[2] = offset_coords[2];

    MPI_Cart_rank(global_comm,offset_coords,&rank_offset);

    // wait for communication
    MPI_Wait(&sreq,&sstat);
    MPI_Wait(&rreq,&rstat);


    // iterate about neighbor borders to determine the neighborship relation
    n_neighbors->at(4) = 0;
    for (int y = 0; y < global_dims[1]; ++y)
    {
        for (int x = 0; x < global_dims[0]; ++x)
        {
            if (
                   ( 
                        ( vertices_rem[4*(x+y*global_dims[0])+2] <= vertices_loc[2] && vertices_loc[2] <  vertices_rem[4*(x+y*global_dims[0])+3] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])+2] <  vertices_loc[3] && vertices_loc[3] <= vertices_rem[4*(x+y*global_dims[0])+3] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])+2] >= vertices_loc[2] && vertices_loc[2] < vertices_rem[4*(x+y*global_dims[0])+3] && 
                          vertices_loc[3] >= vertices_rem[4*(x+y*global_dims[0])+3] )
                   )
               )
            if (
                   ( 
                        ( vertices_rem[4*(x+y*global_dims[0])] <= vertices_loc[0] && vertices_loc[0] <  vertices_rem[4*(x+y*global_dims[0])+1] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])] <  vertices_loc[1] && vertices_loc[1] <= vertices_rem[4*(x+y*global_dims[0])+1] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])] >= vertices_loc[0] && vertices_loc[0] < vertices_rem[4*(x+y*global_dims[0])+1] && 
                          vertices_loc[1] >= vertices_rem[4*(x+y*global_dims[0])+1] )
                   )
               )
            {
                n_neighbors->at(4)++;
                rem_coords[1] = y;
                rem_coords[0] = x;
                MPI_Cart_rank(global_comm,rem_coords,&rem_rank);
                neighbors.push_back(rem_rank);
            }
        }
    }

    // barrier to ensure every process concluded the calculations before overwriting remote borders!
    MPI_Barrier(global_comm);

    // exchange local column information with upper neighbor in Y direction (cart grid)
    MPI_Irecv(vertices_rem                                ,4*global_dims[0]*global_dims[1],mpi_data_type_T,rank_right, 0,global_comm,&rreq);
    MPI_Isend(vertices_rem+4*global_dims[0]*global_dims[1],4*global_dims[0]*global_dims[1],mpi_data_type_T,rank_left,0,global_comm,&sreq);

    // determine the offset in ranks
    offset_coords[0] = 0;
    offset_coords[1] = 0;
    offset_coords[2] = local_coords[2]+1;

    rem_coords[2] = offset_coords[2];

    MPI_Cart_rank(global_comm,offset_coords,&rank_offset);

    // wait for communication
    MPI_Wait(&sreq,&sstat);
    MPI_Wait(&rreq,&rstat);

    // iterate about neighbor borders to determine the neighborship relation
    n_neighbors->at(5) = 0;
    for (int y = 0; y < global_dims[1]; ++y)
    {
        for (int x = 0; x < global_dims[0]; ++x)
        {
            if (
                   ( 
                        ( vertices_rem[4*(x+y*global_dims[0])+2] <= vertices_loc[2] && vertices_loc[2] <  vertices_rem[4*(x+y*global_dims[0])+3] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])+2] <  vertices_loc[3] && vertices_loc[3] <= vertices_rem[4*(x+y*global_dims[0])+3] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])+2] >= vertices_loc[2] && vertices_loc[2] < vertices_rem[4*(x+y*global_dims[0])+3] && 
                          vertices_loc[3] >= vertices_rem[4*(x+y*global_dims[0])+3] )
                   )
               )
            if (
                   ( 
                        ( vertices_rem[4*(x+y*global_dims[0])] <= vertices_loc[0] && vertices_loc[0] <  vertices_rem[4*(x+y*global_dims[0])+1] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])] <  vertices_loc[1] && vertices_loc[1] <= vertices_rem[4*(x+y*global_dims[0])+1] ) ||
                        ( vertices_rem[4*(x+y*global_dims[0])] >= vertices_loc[0] && vertices_loc[0] < vertices_rem[4*(x+y*global_dims[0])+1] && 
                          vertices_loc[1] >= vertices_rem[4*(x+y*global_dims[0])+1] )
                   )
               )
            {
                n_neighbors->at(5)++;
                rem_coords[1] = y;
                rem_coords[0] = x;
                MPI_Cart_rank(global_comm,rem_coords,&rem_rank);
                neighbors.push_back(rem_rank);
            }
        }
    }

    // barrier to ensure every process concluded the calculations before overwriting remote borders!
    MPI_Barrier(global_comm);

    // clear up vertices array
    delete vertices_loc;
    delete vertices_rem;
}

// provide list of neighbors
template<class T, class W> void ALL_Staggered_LB<T,W>::get_neighbors(std::vector<int>& ret)
{
    ret = neighbors;
}
// provide list of neighbors in each direction
template<class T, class W> void ALL_Staggered_LB<T,W>::get_neighbors(int** ret)
{
    *ret = n_neighbors->data();
}

#endif
