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

#ifndef ALL_UNSTRUCTURED_HEADER_INCLUDED
#define ALL_UNSTRUCTURED_HEADER_INCLUDED

/*

    Unstructured load-balancing scheme:

    Requirements: 
        a) cartesian communicator
        b) 2^(dim) vertices describing the domain (in theory could be more,
           we assume a orthogonal grid as start point)
        c) MPI >= 3.0 due to necessary non-blocking collectives


    Method:

        For each of the vertices of a domain a communicator is created, which contains the
        surrounding processes. For one of the vertices, a domain collects the centres of
        gravitiy and the work of the neighboring domains. In a next step a force acting on
        the vertex is computed as:
            F = 1.0 / ( gamma * n * sum(W_i) ) * sum( W_i * x_i )
            with:

                n:      number of neighboring domains
                W_i:    work on domain i
                x_i:    vector pointing from the vertex to 
                        the center of gravity of domain i
                gamma:  correction factor to control the
                        speed of the vertex shift
        
        */

#include <mpi.h>
#include <exception>
#include <vector>
#include <algorithm>
#include <cmath>
#include "ALL_CustomExceptions.hpp"
#include "ALL_Point.hpp"

// T: data type of vertices used in the balancer
// W: data type of the work used in the balancer
template <class T, class W> class ALL_Unstructured_LB
{
    public:
        ALL_Unstructured_LB() {}
        ALL_Unstructured_LB(int d, W w, T g) : dimension(d),
                                         work(w), 
                                         gamma(g)
        {
            // need the lower and upper bounds
            // of the domain for the tensor-based
            // load-balancing scheme
            n_vertices = (int)std::pow(2,dimension);
            vertices = new std::vector<ALL_Point<T>>(n_vertices);
            vertex_rank = new int[n_vertices];
            shifted_vertices = new std::vector<ALL_Point<T>>(n_vertices);
            // global dimension of the system in each coordinate
            global_dims = new int[dimension];
            // coordinates of the local domain in the system
            local_coords = new int[dimension];
            // periodicity of the MPI communicator / system
            periodicity = new int[dimension];

            // array of MPI communicators for each vertex (2 in 1D, 4 in 2D, 8 in 3D)
            communicators = new MPI_Comm[n_vertices];
        }

        ~ALL_Unstructured_LB(); 

        void set_vertices(T*);
        void set_vertices(std::vector<ALL_Point<T>>&);

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
        void get_shifted_vertices(std::vector<ALL_Point<T>>&);

        // neighbors
        // provide list of neighbors
        virtual void get_neighbors(std::vector<int>&);
        // provide list of neighbors in each direction
        virtual void get_neighbors(int**);


    private:
        // dimension of the system
        int dimension;

        // number of vertices (based on dimension)
        int n_vertices;

        // shift cooeficient
        T gamma;

        // work per domain
        W work;

        // vertices
        std::vector<ALL_Point<T>>* vertices; 

        // vertices after load-balancing shift
        std::vector<ALL_Point<T>>* shifted_vertices; 

        // MPI values
        // global communicator
        MPI_Comm global_comm;
        // rank of the local domain in the communicator
        int local_rank;
        // rank(s) in the communicators of the vertices
        int* vertex_rank;
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

        // find neighbors (more sophisticated than for tensor-product
        // variant of LB)
        void find_neighbors();
};

template <class T, class W> ALL_Unstructured_LB<T,W>::~ALL_Unstructured_LB()
{
    if (vertices) delete vertices;
    if (vertex_rank) delete vertex_rank;
    if (shifted_vertices) delete shifted_vertices;
    if (global_dims) delete global_dims;
    if (local_coords) delete local_coords;
    if (periodicity) delete periodicity;
    if (communicators) delete communicators;
}

template <class T, class W> void ALL_Unstructured_LB<T,W>::get_vertices(T* result)
{
    for (int i = 0; i < vertices->size()*dimension; ++i)
    {
        result[i] = vertices->at(i);
    }
}

template <class T, class W> void ALL_Unstructured_LB<T,W>::get_shifted_vertices(T* result)
{
    for (int i = 0; i < vertices->size(); ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
           result[i*dimension+j] = shifted_vertices->at(i).x(j);
        }
    }
}

template <class T, class W> void ALL_Unstructured_LB<T,W>::get_shifted_vertices(std::vector<ALL_Point<T>>& result)
{
    for (int i = 0; i < shifted_vertices->size(); ++i)
    {
        result.at(i) = shifted_vertices->at(i);
    }
}

// set the actual vertices (unsafe due to array copy)
template <class T, class W> void ALL_Unstructured_LB<T,W>::set_vertices(T* v)
{
    for (int i = 0; i < vertices->size()*dimension; ++i)
    {
        vertices->at(i) = v[i];
    }
}

// set the actual vertices 
template <class T, class W> void ALL_Unstructured_LB<T,W>::set_vertices(std::vector<ALL_Point<T>>& v)
{
    // TODO: check if size(v) = 2^dim 
    *vertices = v;
}

// provide list of neighbors
template<class T, class W> void ALL_Unstructured_LB<T,W>::get_neighbors(std::vector<int>& ret)
{
    ret = neighbors;
}
// provide list of neighbors in each direction
template<class T, class W> void ALL_Unstructured_LB<T,W>::get_neighbors(int** ret)
{
    *ret = neighbors.data();
}

template <class T, class W> void ALL_Unstructured_LB<T,W>::setup(MPI_Comm comm)
{
    global_comm = comm;

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

    MPI_Group all;
    MPI_Comm_group(global_comm,&all);

    // check if communicator is cartesian
    int status;
    MPI_Topo_test(global_comm, &status);

    if (status != MPI_CART)
    {
        throw ALL_Invalid_Comm_Type_Exception(
                __FILE__,
                __func__,
                __LINE__,
                "Cartesian MPI communicator required, passed communicator is not cartesian");
    }

    // get the local coordinates, periodicity and global size from the MPI communicator
    MPI_Cart_get(global_comm, dimension, global_dims, periodicity, local_coords);

    // get the local rank from the MPI communicator
    MPI_Cart_rank(global_comm, local_coords, &local_rank);

    // groups required for new communicators
    MPI_Group groups[n_vertices];
    // arrays of processes belonging to group
    int processes[n_vertices][n_vertices];

    // shifted local coordinates to find neighboring processes
    int shifted_coords[dimension];

    /*

    // setup communicators for each of the vertices
    if (dimension == 1)
    {

        // sequence of vertices in 1d
        //
        // 0 - - - 1

        for (int x = 0; x < 2; ++x)
        {
            // group / communicator for vertex 0
            shifted_coords[0] = local_coords[0] - 1 + x;

            MPI_Cart_rank(global_comm, shifted_coords, &(processes[x][0])); 

            shifted_coords[0] = local_coords[0] + x;

            MPI_Cart_rank(global_comm, shifted_coords, &(processes[x][1])); 

            MPI_Group_incl(all, 2, &processes[x][0], &groups[x]);
            MPI_Comm_create(global_comm, groups[x], &communicators[x]);

            // get local rank in vertex communicator
            MPI_Comm_rank(communicators[x],&vertex_rank[x]);
        }
    }            
    else if (dimension == 2)
    {
        // sequence of vertices in 2d
        //
        // 2 - - - 3
        // |       |
        // |       |
        // 0 - - - 1

        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                int vertex = 2 * y + x;

                shifted_coords[0] = local_coords[0] - 1 + x;
                shifted_coords[1] = local_coords[1] - 1 + y;

                MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][0])); 

                shifted_coords[0] = local_coords[0] + x;
                shifted_coords[1] = local_coords[1] - 1 + y;

                MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][1])); 

                shifted_coords[0] = local_coords[0] - 1 + x;
                shifted_coords[1] = local_coords[1] + y;

                MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][2])); 

                shifted_coords[0] = local_coords[0] + x;
                shifted_coords[1] = local_coords[1] + y;

                MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][3])); 

                MPI_Group_incl(all, 4, &processes[vertex][0], &groups[vertex]);
                MPI_Comm_create(global_comm, groups[vertex], &communicators[vertex]);

                // get local rank in vertex communicator
                MPI_Comm_rank(communicators[vertex],&vertex_rank[vertex]);
            }
        }
    }
    else if (dimension == 3)
    {
        // sequence of vertices in 3d
        //
        //    6 - - - 7
        //   /|      /|
        //  / |     / |
        // 4 -2- - 5  3
        // | /     | /
        // |/      |/
        // 0 - - - 1

        for (int z = 0; z < 2; ++z)
        {
            for (int y = 0; y < 2; ++y)
            {
                for (int x = 0; x < 2; ++x)
                {
                    int vertex = 4 * z + 2 * y + x;

                    // group / communicator for vertex 0,2,4,6
                    shifted_coords[0] = local_coords[0] - 1 + x;
                    shifted_coords[1] = local_coords[1] - 1 + y;
                    shifted_coords[2] = local_coords[2] - 1 + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][0])); 

                    shifted_coords[0] = local_coords[0] + x;
                    shifted_coords[1] = local_coords[1] - 1 + y;
                    shifted_coords[2] = local_coords[2] - 1 + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][1])); 

                    shifted_coords[0] = local_coords[0] - 1 + x;
                    shifted_coords[1] = local_coords[1] + y;
                    shifted_coords[2] = local_coords[2] - 1 + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][2])); 

                    shifted_coords[0] = local_coords[0] + x;
                    shifted_coords[1] = local_coords[1] + y;
                    shifted_coords[2] = local_coords[2] - 1 + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][3])); 

                    shifted_coords[0] = local_coords[0] - 1 + x;
                    shifted_coords[1] = local_coords[1] - 1 + y;
                    shifted_coords[2] = local_coords[2] + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][4])); 

                    shifted_coords[0] = local_coords[0] + x;
                    shifted_coords[1] = local_coords[1] - 1 + y;
                    shifted_coords[2] = local_coords[2] + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][5])); 

                    shifted_coords[0] = local_coords[0] - 1 + x;
                    shifted_coords[1] = local_coords[1] + y;
                    shifted_coords[2] = local_coords[2] + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][6])); 

                    shifted_coords[0] = local_coords[0] + x;
                    shifted_coords[1] = local_coords[1] + y;
                    shifted_coords[2] = local_coords[2] + z;

                    MPI_Cart_rank(global_comm, shifted_coords, &(processes[vertex][7])); 

                    MPI_Group_incl(all, 8, &processes[vertex][0], &groups[vertex]);
                    MPI_Comm_create(global_comm, groups[vertex], &communicators[vertex]);

                    // get local rank in vertex communicator
                    MPI_Comm_rank(communicators[vertex],&vertex_rank[vertex]);
                }
            }
        }
        for (int i = 0; i < 8; ++i)
        {
            if (i == local_rank)
            {
                std::cout << "local rank: " << i << " " << std::endl;
                for (int k = 0; k < 8; ++k)
                {
                    std::cout << "group " << k << ": ";
                    for (int j = 0; j < 8; ++j)
                        std::cout << processes[k][j] << " ";
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                MPI_Barrier(MPI_COMM_WORLD);
            }
            else
                MPI_Barrier(MPI_COMM_WORLD);

        }
    }
    else
    {
        throw ALL_Invalid_Argument_Exception(   "ALL_Unstructured.hpp",
                                                "ALL_Unstructured_LB<T,W>::setup(int)",
                                                257,
                                                "Unsupported number of dimensions.");
    }
    */

    int dim_vert[dimension];

    for (int d = 0; d < dimension; ++d)
    {
        dim_vert[d] = global_dims[d];
    }

    MPI_Comm tmp_comm;
    int own_vertex;

    for (int i = 0; i < n_vertices; ++i)
        communicators[i] = MPI_COMM_SELF;

    for (int iz = 0; iz < dim_vert[2]; ++iz)
    {
        for (int iy = 0; iy < dim_vert[1]; ++iy)
        {
            for (int ix = 0; ix < dim_vert[0]; ++ix)
            {   
                if (ix == ( ( local_coords[0] + 0 ) % global_dims[0])  &&
                    iy == ( ( local_coords[1] + 0 ) % global_dims[1])  &&
                    iz == ( ( local_coords[2] + 0 ) % global_dims[2]) )
                    own_vertex = 0;
                else if (ix == ( ( local_coords[0] + 1 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 0 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 0 ) % global_dims[2]) )
                    own_vertex = 1;
                else if (ix == ( ( local_coords[0] + 0 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 1 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 0 ) % global_dims[2]) )
                    own_vertex = 2;
                else if (ix == ( ( local_coords[0] + 1 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 1 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 0 ) % global_dims[2]) )
                    own_vertex = 3;
                else if (ix == ( ( local_coords[0] + 0 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 0 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 1 ) % global_dims[2]) )
                    own_vertex = 4;
                else if (ix == ( ( local_coords[0] + 1 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 0 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 1 ) % global_dims[2]) )
                    own_vertex = 5;
                else if (ix == ( ( local_coords[0] + 0 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 1 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 1 ) % global_dims[2]) )
                    own_vertex = 6;
                else if (ix == ( ( local_coords[0] + 1 ) % global_dims[0])  &&
                         iy == ( ( local_coords[1] + 1 ) % global_dims[1])  &&
                         iz == ( ( local_coords[2] + 1 ) % global_dims[2]) )
                    own_vertex = 7;
                else 
                    own_vertex = -1;

                // if vertex belongs to local domain, join communicator, otherwise ignore it
                int result;
                if (own_vertex >= 0)
                {
                    vertex_rank[own_vertex] = 7 - own_vertex;
                    result = MPI_Comm_split(global_comm,1,vertex_rank[own_vertex],&communicators[own_vertex]);
                    MPI_Comm_rank(communicators[own_vertex],&vertex_rank[own_vertex]);
                }
                else
                {
                    result = MPI_Comm_split(global_comm,0,0,&tmp_comm);
                }
                if (result != MPI_SUCCESS)
                {
                    std::cout << "RANK: " << local_rank << " ERROR in MPI_Comm_Split" << std::endl;
                }
            }
        }
    }
    for (int vertex = 0; vertex < n_vertices; ++vertex)
        MPI_Comm_rank(communicators[vertex],&vertex_rank[vertex]);

}

// TODO: periodic boundary conditions (would require size of the system)
//       for now: do not change the vertices along the edge of the system

template <class T, class W> void ALL_Unstructured_LB<T,W>::balance()
{
    // geometrical center and work of each neighbor for each vertex
    // work is cast to vertex data type, since in the end a shift
    // of the vertex is computed
    T vertex_info[n_vertices*n_vertices*(dimension+2)];
    T center[dimension];

    // local geometric center
    T local_info[(dimension+2) * n_vertices];

    for (int i = 0; i < n_vertices*n_vertices*(dimension+2); ++i)
        vertex_info[i] = -1;

    for (int d = 0; d < (dimension+2) * n_vertices; ++d)
        local_info[d] = (T)0;

    for (int d = 0; d < dimension; ++d)
        center[d] = (T)0;

    // compute geometric center
    for (int v = 0; v < n_vertices; ++v)
    {
        for (int d = 0; d < dimension; ++d)
        {
            center[d] += ( (vertices->at(v)).x(d) / (T)n_vertices );
        }
        local_info[v * (dimension+2) + dimension] = (T)work;
    }
    for (int v = 0; v < n_vertices; ++v)
    {
        for (int d = 0; d < dimension; ++d)
        {
            local_info[v * (dimension+2) + d] = center[d] - (vertices->at(v)).x(d);
        }
    }

    // compute maximum distance of a vertex to the plane of the neighboring three vertices
    auto distance = [=] ( std::vector<ALL_Point<T>>* verts, int p, int a, int b, int c, int rank )
    {
        // vectors spanning the plane from vertex 'a'
        T vb[3];
        T vc[3];
        
        for (int i = 0; i < 3; ++i)
        {
            vb[i] = verts->at(b).x(i) - verts->at(a).x(i);
            vc[i] = verts->at(c).x(i) - verts->at(a).x(i);
        }
        
        // normal vector of plane
        T n[3];

        n[0] = vb[1] * vc[2] - vb[2] * vc[1];        
        n[1] = vb[2] * vc[0] - vb[0] * vc[2];        
        n[2] = vb[0] * vc[1] - vb[1] * vc[0];

        // length of normal vector
        T dn = sqrt
            (
                n[0] * n[0] +
                n[1] * n[1] +
                n[2] * n[2]
            );

        // distance from origin
        T d = ( n[0] * verts->at(a).x(0) +
                n[1] * verts->at(a).x(1) +
                n[2] * verts->at(a).x(2)); 

        /*
        if (rank == 4)
        {
            std::cout   << p << " " << std::endl
                        << " A: " << verts->at(a).x(0) << " " << verts->at(a).x(1) << " " << verts->at(a).x(2) << " " << std::endl
                        << " B: " << verts->at(b).x(0) << " " << verts->at(b).x(1) << " " << verts->at(b).x(2) << " " << std::endl
                        << " C: " << verts->at(c).x(0) << " " << verts->at(c).x(1) << " " << verts->at(c).x(2) << " " << std::endl
                        << " b: " << vb[0] << " " << vb[1] << " " << vb[2] << std::endl
                        << " c: " << vc[0] << " " << vc[1] << " " << vc[2] << std::endl
                        << " n: " << n[0] << " " << n[1] << " " << n[2] << std::endl
                        << " |n|: " << dn << std::endl
                        << " d: " << d << std::endl
                        << " dp: " << std::abs (
                                                (
                                                   n[0] * verts->at(p).x(0)
                                                   + n[1] * verts->at(p).x(1)
                                                   + n[2] * verts->at(p).x(2)
                                                   - d
                                                ) / dn
                                              ) << std::endl;
        }
        */

        // compute distance of test vertex to plane
        return std::abs (
                    (
                          n[0] * verts->at(p).x(0) 
                        + n[1] * verts->at(p).x(1)
                        + n[2] * verts->at(p).x(2)
                        - d
                    ) / dn
               );

    };

    // compute for each vertex the maximum movement towards the center

    local_info[ 0 * (dimension+2) + dimension+1 ] =
        distance( vertices, 0, 1, 2, 4, local_rank);

    local_info[ 1 * (dimension+2) + dimension+1 ] =
        distance( vertices, 1, 0, 3, 5, local_rank);

    local_info[ 2 * (dimension+2) + dimension+1 ] =
        distance( vertices, 2, 3, 0, 6, local_rank);

    local_info[ 3 * (dimension+2) + dimension+1 ] =
        distance( vertices, 3, 2, 1, 7, local_rank);

    local_info[ 4 * (dimension+2) + dimension+1 ] =
        distance( vertices, 4, 5, 6, 0, local_rank);

    local_info[ 5 * (dimension+2) + dimension+1 ] =
        distance( vertices, 5, 4, 7, 1, local_rank);

    local_info[ 6 * (dimension+2) + dimension+1 ] =
        distance( vertices, 6, 7, 4, 2, local_rank);

    local_info[ 7 * (dimension+2) + dimension+1 ] =
        distance( vertices, 7, 6, 5, 3, local_rank);

    /*
    if (local_rank == 4)
    {
        std::cout << "max dists: " 
                  << local_info[ 0 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 1 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 2 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 3 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 4 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 5 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 6 * (dimension+2) + dimension+1 ] << " "
                  << local_info[ 7 * (dimension+2) + dimension+1 ] << " "
                  << work << std::endl;
    }
    */


    // exchange information with all vertex neighbors
    MPI_Request request[n_vertices];
    MPI_Status status[n_vertices];

    for (int v = 0; v < n_vertices; ++v)
    {
        MPI_Iallgather(&local_info[v*(dimension+2)],dimension+2,mpi_data_type_T,
                       &vertex_info[v*n_vertices*(dimension+2)],dimension+2,mpi_data_type_T,
                       communicators[v],&request[v]);
    }
    MPI_Waitall(n_vertices,request,status);

    // compute new position for vertex 7 (if not periodic)
    T total_work = (T)0;
    T shift_vectors[dimension * n_vertices];

    for (int v = 0; v < n_vertices; ++v)
    {
        for (int d = 0; d < dimension; ++d)
        {
            shift_vectors[v * dimension + d] = (T)0;
        }
    }

    // get total work
    for (int v = 0; v < n_vertices; ++v)
    {
        if (vertex_info[(n_vertices-1)*n_vertices+v*(dimension+2)+dimension] > 1e-6)
            total_work += vertex_info[(n_vertices-1)*n_vertices+v*(dimension+2)+dimension];
    }

    // determine shortest distance between vertex and a neighbor 
    T cdist = std::max( 
                        1e-6,
                        vertex_info[(n_vertices-1)*(n_vertices*(dimension+2))+0*(dimension+2)+dimension+1]
                      );

    for (int v = 1; v < n_vertices; ++v)
    {
        if ( vertex_info[(n_vertices-1)*(n_vertices*(dimension+2))+v*(dimension+2)+dimension+1] > 0 )
        {
            cdist = std::min
                    (
                        cdist,
                        vertex_info[(n_vertices-1)*(n_vertices*(dimension+2))+v*(dimension+2)+dimension+1]
                    );
        }
    }
    cdist *= 0.49;

    cdist = 0.5;

    if (std::abs(total_work) >= 1e-6)
    {
        for (int i = 0; i < n_vertices; ++i)
        {
            for (int d = 0; d < dimension; ++d)
            {
                if (local_coords[d] != global_dims[d] - 1)
                {
                    //if ( vertex_info[(n_vertices-1)*(n_vertices*(dimension+2))+i*(dimension+2)+dimension] - ( total_work / (T)n_vertices )  > 0)                    
                    //{
                        shift_vectors[(n_vertices-1)*dimension + d] += 
                            ( vertex_info[(n_vertices-1)*(n_vertices*(dimension+2))+i*(dimension+2)+dimension] - ( total_work / (T)n_vertices ) ) *
                            vertex_info[(n_vertices-1)*(n_vertices*(dimension+2))+i*(dimension+2)+d]  
                            / ( (T)2.0 * gamma * total_work * (T)n_vertices);
                    //}
                }
            }
        }
        // distance of shift
        T shift = (T)0;
        for (int d = 0; d < dimension; ++d)
            shift +=  shift_vectors[(n_vertices-1)*dimension + d] * shift_vectors[(n_vertices-1)*dimension + d];
        shift = sqrt(shift);

        if (std::abs(shift) >= 1e-6 && shift > cdist)
        {
            // normalize length of shift vector to half the distance to the nearest center
            for (int d = 0; d < dimension; ++d)
                shift_vectors[(n_vertices-1)*dimension + d] *= cdist / shift;
        }

    }

    for (int i = 0; i < n_vertices; ++i)
    {
        MPI_Ibcast(&shift_vectors[i*dimension],dimension,mpi_data_type_T,0,communicators[i],&request[i]);
    }
    MPI_Waitall(n_vertices,request,status);

    shifted_vertices = vertices;
    /*
    if (local_rank == 7)
    {
        std::cout << "center: ";
        for (int d = 0; d < dimension; ++d)
        {
            std::cout << center[d] << " ";
        }
        std::cout << std::endl;
    }
    */
    for (int v = 0; v < n_vertices; ++v)
    {
        //if (local_rank == 4) std::cout << "vertex " << v << ": ";
        for (int d = 0; d < dimension; ++d)
        {
            shifted_vertices->at(v).set_coordinate(d, vertices->at(v).x(d) + shift_vectors[v * dimension + d]);
            if (shifted_vertices->at(v).x(d) != shifted_vertices->at(v).x(d))
            {
                std::cout << local_rank << " " << v << " " << d << std::endl;
                MPI_Abort(MPI_COMM_WORLD,-200);
            }
            //if (local_rank == 4) std::cout << shifted_vertices->at(v).x(d) << " " << shift_vectors[v * dimension + d] << " ";
        }
        //if (local_rank == 4) std::cout << std::endl;
    }
}

#endif

