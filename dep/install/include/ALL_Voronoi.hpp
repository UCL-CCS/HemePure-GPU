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

#ifndef ALL_VORONOI_HEADER_INCLUDED
#define ALL_VORONOI_HEADER_INCLUDED

// number of maximum neighboring Voronoi cells
#define ALL_VORONOI_MAX_NEIGHBORS 32
// number of subblock division for Voronoi cell creation
#define ALL_VORONOI_SUBBLOCKS 20 

// depth used to find neighbor cells
#define ALL_VORONOI_NEIGHBOR_SEARCH_DEPTH 2

#ifdef ALL_VORONOI
#include <mpi.h>
#include <exception>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include "ALL_CustomExceptions.hpp"
#include "ALL_Point.hpp"

#include "voro++.hh"

// T: data type of vertices used in the balancer
// W: data type of the work used in the balancer
template <class T, class W> class ALL_Voronoi_LB
{
    public:
        ALL_Voronoi_LB() {}
        ALL_Voronoi_LB(int d, W w, T g) : dimension(d),
                                         work(w), 
                                         gamma(g)
        {
            // only need the generator point
            vertices = new T[2*dimension];
            shifted_vertices = new T[dimension];
            // set size for system size array
            sys_size.resize(6);

            loadbalancing_step = 0;
        }

        ~ALL_Voronoi_LB(); 

        void set_vertices(T*);
        void set_vertices(std::vector<ALL_Point<T>>&);

        voro::voronoicell vc;

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
        // getter for shifted vertices
        void get_shifted_vertices(std::vector<ALL_Point<T>>&);

        // neighbors
        // provide list of neighbors
        virtual void get_neighbors(std::vector<int>&);
        // provide list of neighbors in each direction
        virtual void get_neighbors(int**);
        // return generator points of the neighboring cells (to be able to recreate
        // local and neighboring cells)
        virtual void get_neighbor_vertices(std::vector<T>&);

        // set system size for periodic corrections
        void set_sys_size(std::vector<T>&);

        void set_step(int s) { loadbalancing_step = s; }

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

        
        // type for MPI communication
        MPI_Datatype mpi_data_type_T;
        MPI_Datatype mpi_data_type_W;

        // list of neighbors
        std::vector<int> neighbors;
        int n_neighbors[1];

        // generator points of the neighbor cells
        std::vector<T> neighbor_vertices;

        // collection of generator points
        std::vector<T> generator_points;

        // size of system (for periodic boxes / systems)
        std::vector<T> sys_size;

        // number of ranks in global communicator
        int n_domains;

        int loadbalancing_step;
};

template <class T, class W> ALL_Voronoi_LB<T,W>::~ALL_Voronoi_LB()
{
    if (vertices) delete vertices;
    if (shifted_vertices) delete shifted_vertices;
}

template <class T, class W> void ALL_Voronoi_LB<T,W>::get_vertices(T* result)
{
    for (int i = 0; i < dimension; ++i)
    {
        result[i] = vertices[i];
    }
}

template <class T, class W> void ALL_Voronoi_LB<T,W>::get_shifted_vertices(T* result)
{
    for (int i = 0; i < dimension; ++i)
    {
        result[i] = shifted_vertices[i];
    }
}

// getter for shifted vertices
template <class T, class W> void ALL_Voronoi_LB<T,W>::get_shifted_vertices(std::vector<ALL_Point<T>>& result)
{
    ALL_Point<T> p(
                    3,
                    { shifted_vertices[0],
                      shifted_vertices[1],
                      shifted_vertices[2] }
                  ); 
    result.clear();
    result.push_back(p);
}

template <class T, class W> void ALL_Voronoi_LB<T,W>::get_shifted_vertices(std::vector<T>& result)
{
    for (int i = 0; i < dimension; ++i)
    {
        result.at(i) = shifted_vertices[i];
    }
}

// set the actual vertices (unsafe due to array copy)
template <class T, class W> void ALL_Voronoi_LB<T,W>::set_vertices(T* v)
{
    for (int i = 0; i < 2*dimension; ++i)
    {
        vertices[i] = v[i];
    }
}

// set the actual vertices
template <class T, class W> void ALL_Voronoi_LB<T,W>::set_vertices(std::vector<ALL_Point<T>>& _v)
{
    for (int v = 0; v < 2; ++v)
        for (int i = 0; i < dimension; ++i)
        {
            vertices[dimension * v + i] = _v.at(v).x(i);
        }
}

template <class T, class W> void ALL_Voronoi_LB<T,W>::set_sys_size(std::vector<T>& ss)
{
    for (auto i = 0; i < sys_size.size(); ++i)
    {
        sys_size.at(i) = ss.at(i);
    }
}

// setup routine for the tensor-based load-balancing scheme
// requires: 
//              global_comm (int): cartesian MPI communicator, from
//                                 which separate sub communicators
//                                 are derived in order to represent
//                                 each plane of domains in the system
template <class T, class W> void ALL_Voronoi_LB<T,W>::setup(MPI_Comm comm)
{
    // store global communicator
    global_comm = comm; 

    // no special communicator required

    // create array to store information about generator points
    // TODO: hierarchical scheme or better preselection which
    //       generator points are required for correct creation
    //       of the local cell -> large-scale Voronoi grid creation
    //                            is too expensive to be done
    //                            every step
    MPI_Comm_size(global_comm, &n_domains);
    MPI_Comm_rank(global_comm, &local_rank);
    generator_points.resize(n_domains*(2*dimension+1));

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

template<class T, class W> void ALL_Voronoi_LB<T,W>::balance()
{
    // collect local information in array
    int info_length = 2*dimension+1;
    std::vector<T> local_info(info_length);
    
    for (auto d = 0; d < 2*dimension; ++d)
    {
        local_info.at(d) = vertices[d];
    }
    local_info.at(2*dimension) = (T)work;

    
    MPI_Allgather(local_info.data(), info_length, mpi_data_type_T, 
                  generator_points.data(), info_length, mpi_data_type_T,
                  global_comm);

    // create Voronoi-cells
    // TODO: check if system is periodic or not -> true / false!
    voro::container con_old(
                sys_size.at(0),
                sys_size.at(1),
                sys_size.at(2),
                sys_size.at(3),
                sys_size.at(4),
                sys_size.at(5),
                ALL_VORONOI_SUBBLOCKS,
                ALL_VORONOI_SUBBLOCKS,
                ALL_VORONOI_SUBBLOCKS,
                false,
                false,
                false,
                n_domains+10 
            );

    // add generator points to container
    for (auto d = 0; d < n_domains; ++d)
    {
        con_old.put(
                    d,
                    generator_points.at(info_length*d),
                    generator_points.at(info_length*d+1),
                    generator_points.at(info_length*d+2)
               );
    }

    // print old voronoi cell structure 
/*
    std::ostringstream ss_local_gp_pov;
    ss_local_gp_pov << "voronoi/generator_points_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".pov";
    std::ostringstream ss_local_gp_gnu;
    ss_local_gp_gnu << "voronoi/generator_points_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".gnu";
    std::ostringstream ss_local_vc_pov;
    ss_local_vc_pov << "voronoi/voronoi_cells_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".pov";
    std::ostringstream ss_local_vc_gnu;
    ss_local_vc_gnu << "voronoi/voronoi_cells_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".gnu";

    loadbalancing_step++;

    if (local_rank == 0)
    {
        con_old.draw_particles(ss_local_gp_gnu.str().c_str());
        con_old.draw_cells_gnuplot(ss_local_vc_gnu.str().c_str());
        con_old.draw_particles_pov(ss_local_gp_pov.str().c_str());
        con_old.draw_cells_pov(ss_local_vc_pov.str().c_str());
    }
*/

    // vector to store neighbor information
    std::vector<double> cell_vertices;
    voro::voronoicell_neighbor c;

    // generate loop over all old generator points
    voro::c_loop_all cl_old(con_old);
    
    int pid;
    double x, y, z, r;
    // reset list of neighbors
    neighbors.clear();

    std::map<int,double> neig_area;

    for (auto i = 0; i < 1; ++i)
    {
        // find next neighbors
        std::vector<std::vector<int>> next_neighbors(std::max((int)neighbors.size(),1));
        std::vector<std::vector<double>> neighbors_area(std::max((int)neighbors.size(),1));
        int idx = 0;
        if (cl_old.start())
        {
            do
            {
                // compute only local cell
                cl_old.pos(pid,x,y,z,r);
                if (i == 0)
                {
                    if (pid == local_rank)
                    {
                        con_old.compute_cell(c,cl_old);
                        c.neighbors(next_neighbors.at(idx));
                        c.face_areas(neighbors_area.at(idx));
                        for (int j = 0; j < next_neighbors.at(idx).size(); ++j)
                            neig_area.insert(std::pair<int,double>(next_neighbors.at(idx).at(j),neighbors_area.at(idx).at(j)));
                        idx++;
                        if (idx == neighbors.size()) break;
                    }
                }
                else
                {
                    if (std::count(neighbors.begin(), neighbors.end(), pid) == 1)
                    {
                        con_old.compute_cell(c,cl_old);
                        c.neighbors(next_neighbors.at(idx));
                        idx++;
                        if (idx == neighbors.size()) break;
                    }
                }
            }
            while(cl_old.inc());
        }
        for (auto it = next_neighbors.begin(); it != next_neighbors.end(); ++it)
        {
            neighbors.insert(neighbors.begin(), it->begin(), it->end());
        }

        std::sort(neighbors.begin(),neighbors.end());
        auto uniq_it = std::unique(neighbors.begin(), neighbors.end());
        neighbors.resize( std::distance(neighbors.begin(), uniq_it) );
        neighbors.erase(std::remove(neighbors.begin(),neighbors.end(),local_rank),neighbors.end());
        neighbors.erase(std::remove_if(neighbors.begin(),neighbors.end(),[](int x){return x < 0;}),neighbors.end());
    }



    std::vector<double> volumes(neighbors.size());
    std::vector<double> surfaces(neighbors.size());

    for (int i = 0; i < neighbors.size(); ++i)
    {
        auto it = neig_area.find(neighbors.at(i));
        surfaces.at(i) = it->second;
    }

    double local_volume;
    // get volumes of each neighbor cell
    if (cl_old.start())
    {
        do
        {
            cl_old.pos(pid,x,y,z,r);
            auto it = std::find(neighbors.begin(),neighbors.end(),pid);

            if (it != neighbors.end())
            {
                int idx = std::distance(neighbors.begin(), it);
                con_old.compute_cell(c,cl_old);
                volumes.at(idx) = c.volume();
            }
            if (pid == local_rank)
            {
                con_old.compute_cell(c,cl_old);
                local_volume = c.volume();
            }
        }
        while(cl_old.inc());
    }

    MPI_Barrier(global_comm);
    if(local_rank == 0) std::cout << "still sane?" << std::endl;

    /* 
    for (int i = 0; i < 25; ++i)
    {
        if (local_rank == i)
        {
            std::cout << local_rank << ": ";
            for (auto n : neighbors)
                std::cout << " " << n << " ";
            std::cout << "| " << neighbors.size() << std::endl;
            for (auto s : surfaces)
                std::cout << " " << s << " ";
            std::cout << std::endl;
        }
        MPI_Barrier(global_comm);
    }
    */

    // compute shift
    std::vector<T> shift(dimension, 0.0);
    T norm;


    T work_load = (T)local_info.at(info_length-1);
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
    {
        int neighbor = *it;
        work_load += generator_points.at(info_length * neighbor + info_length - 1);
    }

    T max_diff = 20.0;

    for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
    {
        int neighbor = *it;
        std::vector<T> diff(dimension);
        bool correct = true;
        for (int d = 0; d < dimension && correct; ++d)
        {
            diff.at(d) = generator_points.at(info_length * neighbor + dimension + d) - local_info.at(d);
            //diff.at(d) = generator_points.at(info_length * neighbor + d) - local_info.at(d);

            if (diff.at(d) > 0.5 * (sys_size[2*d+1] - sys_size[2*d]))
            {
                diff.at(d) -= (sys_size[2*d+1] - sys_size[2*d]);
                correct = false;
            }
            else if (diff.at(d) < -0.5 * (sys_size[2*d+1] - sys_size[2*d]))
            {
                diff.at(d) += (sys_size[2*d+1] - sys_size[2*d]);
                correct = false;
            }

        }
        max_diff = std::min(max_diff,sqrt(diff.at(0)*diff.at(0)+diff.at(1)*diff.at(1)+diff.at(2)*diff.at(2)));
        if (correct)
        {
            /*
            // normalize direction vector
            T norm_diff = diff.at(0) * diff.at(0) + diff.at(1) * diff.at(1) + diff.at(2) * diff.at(2);
            norm_diff = sqrt(norm_diff);
            for (int d = 0; d < dimension; ++d)
                diff.at(d) /= norm_diff;
            */
            // compute difference in work load
            T work_diff = (T)0.0;
            work_diff = ( generator_points.at(info_length * neighbor + info_length - 1) - local_info.at(info_length-1) );
            //if (work_diff < 0.0) work_diff = 0.0;

            // compute work density between processess
            //T work_density = ( generator_points.at(4 * neighbor + 3) + local_info.at(3) ) /
            //                 ( volumes.at(std::distance(neighbors.begin(),it)) + local_volume );
            T work_density = ( generator_points.at(info_length * neighbor + info_length - 1) + local_info.at(info_length-1) );

            //if (work_diff < 0.0) work_diff *= 0.05;
            if (work_density < 1e-6) 
                work_diff = (T)0;
            else
                work_diff /= work_density;
            for (int d = 0; d < dimension; ++d)
            {
                //shift.at(d) += 0.5 * work_diff / work_density * surfaces.at(std::distance(neighbors.begin(),it)) * diff.at(d);
                shift.at(d) += 0.5 * work_diff * diff.at(d) / gamma;
            }
        }
    }

    norm = sqrt ( shift.at(0) * shift.at(0) + shift.at(1) * shift.at(1) + shift.at(2) * shift.at(2) );

    /*
    for (int i = 0; i < 25; ++i)
    {
        if (local_rank == i)
        {
            std::cout << local_rank << ": ";
            for (auto n : local_info)
                std::cout << " " << n << " ";
            std::cout << " | ";
            for (auto n : shift)
                std::cout << " " << n << " ";
            std::cout << std::endl;
        }
        MPI_Barrier(global_comm);
    }
    */

    T scale = 1.0;

   if (norm > 0.45 * max_diff)
        scale = 0.45 * max_diff / norm;
        

    // to find new neighbors
    for (int d = 0; d < dimension; ++d)
    {
        local_info.at(d) += scale * shift.at(d);
        
        // periodic correction of points
        /*
        local_info.at(d) = (local_info.at(d) < sys_size.at(2*d))
                                ?(local_info.at(d) + (sys_size.at(2*d+1) - sys_size.at(2*d)))
                                : ((local_info.at(d) >= sys_size.at(2*d+1))
                                    ?(local_info.at(d) - (sys_size.at(2*d+1) - sys_size.at(2*d)))
                                    :local_info.at(d));
        */
        // periodic correction of points
        local_info.at(d) = (local_info.at(d) < sys_size.at(2*d))
                                ?sys_size.at(2*d)+1.0
                                : ((local_info.at(d) >= sys_size.at(2*d+1))
                                    ?sys_size.at(2*d+1)-1.0
                                    :local_info.at(d));
    }

    MPI_Allgather(local_info.data(), info_length, mpi_data_type_T, 
                  generator_points.data(), info_length, mpi_data_type_T,
                  global_comm);

    // create Voronoi-cells
    // TODO: check if system is periodic or not -> true / false!
    voro::container con_new(
                sys_size.at(0),
                sys_size.at(1),
                sys_size.at(2),
                sys_size.at(3),
                sys_size.at(4),
                sys_size.at(5),
                ALL_VORONOI_SUBBLOCKS,
                ALL_VORONOI_SUBBLOCKS,
                ALL_VORONOI_SUBBLOCKS,
                false,
                false,
                false,
                n_domains+10 
            );


    // add generator points to container
    for (int d = 0; d < n_domains; ++d)
    {
        con_new.put(
                    d,
                    generator_points.at(info_length*d),
                    generator_points.at(info_length*d+1),
                    generator_points.at(info_length*d+2)
               );
    }

/*
    std::ostringstream ss_local_gp_pov2;
    ss_local_gp_pov2 << "voronoi/generator_points_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".pov";
    std::ostringstream ss_local_gp_gnu2;
    ss_local_gp_gnu2 << "voronoi/generator_points_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".gnu";
    std::ostringstream ss_local_vc_pov2;
    ss_local_vc_pov2 << "voronoi/voronoi_cells_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".pov";
    std::ostringstream ss_local_vc_gnu2;
    ss_local_vc_gnu2 << "voronoi/voronoi_cells_" << std::setw(7) << std::setfill('0') 
             << loadbalancing_step << ".gnu";

    loadbalancing_step++;

    if (local_rank == 0)
    {
        con_new.draw_particles(ss_local_gp_gnu2.str().c_str());
        con_new.draw_cells_gnuplot(ss_local_vc_gnu2.str().c_str());
        con_new.draw_particles_pov(ss_local_gp_pov2.str().c_str());
        con_new.draw_cells_pov(ss_local_vc_pov2.str().c_str());
    }
*/

    for (int d = 0; d < dimension; ++d)
       shifted_vertices[d] = local_info.at(d);

    // compute new neighboring cells and generator points

    // generate loop over all new generator points
    voro::c_loop_all cl_new(con_new);
    
    // reset list of neighbors
    neighbors.clear();

    for (auto i = 0; i < ALL_VORONOI_NEIGHBOR_SEARCH_DEPTH; ++i)
    {
        // find next neighbors
        std::vector<std::vector<int>> next_neighbors(std::max((int)neighbors.size(),1));
        int idx = 0;
        if (cl_new.start())
        {
            do
            {
                // compute next voronoi cell
                cl_new.pos(pid,x,y,z,r);
                if (i == 0)
                {
                    if (pid == local_rank)
                    {
                        con_new.compute_cell(c,cl_new);
                        c.neighbors(next_neighbors.at(idx));
                        idx++;
                        if (idx == neighbors.size()) break;
                    }
                }
                else
                {
                    if (std::count(neighbors.begin(), neighbors.end(), pid) == 1)
                    {
                        con_new.compute_cell(c,cl_new);
                        c.neighbors(next_neighbors.at(idx));
                        idx++;
                        if (idx == neighbors.size()) break;
                    }
                }
            }
            while(cl_new.inc());
        }
        for (auto it = next_neighbors.begin(); it != next_neighbors.end(); ++it)
        {
            neighbors.insert(neighbors.begin(), it->begin(), it->end());
        }

        std::sort(neighbors.begin(),neighbors.end());
        auto uniq_it = std::unique(neighbors.begin(), neighbors.end());
        neighbors.resize( std::distance(neighbors.begin(), uniq_it) );
        neighbors.erase(std::remove(neighbors.begin(),neighbors.end(),local_rank),neighbors.end());
        neighbors.erase(std::remove_if(neighbors.begin(),neighbors.end(),[](int x){return x < 0;}),neighbors.end());
    }

    // determine number of neighbors
    n_neighbors[0] = neighbors.size();


    // clear old neighbor vertices
    neighbor_vertices.clear();

    /*
    // find vertices of neighbors
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
    {
        // reset list of generator points in loop
        if (cl_new.start())
        {
            do
            {
                if (cl_new.pid() == *it)
                {
                    con_new.compute_cell(c,cl_new);
                    c.vertices(cell_vertices);
                    for (auto jt = cell_vertices.begin(); jt != cell_vertices.end(); ++jt)
                        neighbor_vertices.push_back(*jt);
                    break;
                }
            }
            while(cl_new.inc());
        }
    }
    */
    for (auto n : neighbors)
    {
        for (int i = 0; i < dimension; ++i)
            neighbor_vertices.push_back(generator_points.at(info_length * n + i));
    }

/*
    for (int i = 0; i < 25; ++i)
    {
        if (local_rank == i)
        {
            std::cout << local_rank << ": ";
            for (auto n : neighbors)
                std::cout << " " << n << " ";
            std::cout << "| " << neighbors.size() << std::endl;
            for (int n = 0; n < neighbor_vertices.size(); ++n)
            {
                std::cout << neighbor_vertices.at(n) << " ";
                if (n%3 == 2) std::cout << "| ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(global_comm);
    }
*/

}

// provide list of neighbors
template<class T, class W> void ALL_Voronoi_LB<T,W>::get_neighbors(std::vector<int>& ret)
{
    ret = neighbors;
}
// provide list of neighbors in each direction
template<class T, class W> void ALL_Voronoi_LB<T,W>::get_neighbors(int** ret)
{
    *ret = n_neighbors;
}

// provide list of neighbor vertices
template<class T, class W> void ALL_Voronoi_LB<T,W>::get_neighbor_vertices(std::vector<T>& ret)
{
    ret = neighbor_vertices;
}


#endif
#endif
