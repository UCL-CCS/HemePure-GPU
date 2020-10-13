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

#ifndef ALL_POINT_HEADER_INC
#define ALL_POINT_HEADER_INC

#include <cmath>
#include <list> 
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "ALL_CustomExceptions.hpp"

template <class T> class ALL_Point;
template <typename T> std::ostream& operator<< (std::ostream&, const ALL_Point<T>&);

template <class T> class ALL_Point
{
    public:
        // empty constructor
        ALL_Point() : dimension(0) {}
        // constructor for empty list with given length
        ALL_Point(const int d) : dimension(d) {coordinates.resize(d); weight = (T)1;}
        // constructor for ALL_Point with given coordinates (not secure)
        ALL_Point(const int,const T*);
        // constructor for ALL_Point with given coordinates (std::list)
        ALL_Point(const int,const std::list<T>&);
        // constructor for ALL_Point with given coordinates (std::vector)
        ALL_Point(const std::vector<T>&);
        // constructor for ALL_Point with given coordinates with weight (not secure)
        ALL_Point(const int,const T*, const T);
        // constructor for ALL_Point with given coordinates with weight (std::list)
        ALL_Point(const int,const std::list<T>&, const T);
        // constructor for ALL_Point with given coordinates with weight (std::vector)
        ALL_Point(const std::vector<T>&, const T);
        // destructor
        ~ALL_Point<T>() {};

        // TODO: return values to check if operation was successful?
        void set_dimension(const int d) {dimension = d; coordinates.resize(d);}
        // set weight of point
        void set_weight(const T w) {weight = w;}
        // set coordinates with an array (not secure)
        void set_coordinates(const T*);
        // set coordinates with a std::list
        void set_coordinates(const std::list<T>&);
        // set coordinates with a std::vector
        void set_coordinates(const std::vector<T>&);

        // get a single coordinate
        T get_coordinate(const int) const;
        // get weight
        T get_weight() const {return weight;}
        // set a single coordinate
        void set_coordinate(const int,const T&);

        // get dimension of the point
        int get_dimension() const {return dimension;}

        // shorter version of getter for a coordinate
        T x(int) const;

        // compute distances
        // euclidean distance (two-norm)
        double d(ALL_Point<T> p);
        // manhatten / city-block distance (one-norm)
        double d_1(ALL_Point<T> p);

    private:
        // TODO: check how to allocate dynamically?
        // array containg the coordinates
        // std::vector<T>* coordinates;

        // dimension of the ALL_Point
        int dimension;
        // array containg the coordinates
        std::vector<T> coordinates;
        // weight (e.g. number of interactions, sites in case of block-based
        // load-balancing)
        T weight;
};

template <class T> ALL_Point<T>::ALL_Point(const int d, const T* data) : ALL_Point<T>(d)
{
    // copy each element of the array into the vector (insecure, no boundary checks for data!)
    for (auto i = 0; i < d; ++i)
        coordinates.at(i) = data[i];
}

template <class T> ALL_Point<T>::ALL_Point(const int d, const std::list<T>& data) 
    : ALL_Point<T>(d)
{
    // copy the contents of the list data to the coordinates vector
    coordinates.insert(coordinates.begin(),data.begin(),data.end());
}

template <class T> ALL_Point<T>::ALL_Point(const std::vector<T>& data)
{
    // initialize the coordinates vector with the data vector
    coordinates.insert(coordinates.begin(),data.begin(),data.end());
    // update the dimension with the size of the data vector
    dimension = data.size();
}

template <class T> ALL_Point<T>::ALL_Point(const int d, const T* data, const T w) 
    : ALL_Point<T>(d,data)
{
    weight = w;
}

template <class T> ALL_Point<T>::ALL_Point(const int d, const std::list<T>& data, const T w) 
    : ALL_Point<T>(d,data)
{
    weight = w;
}

template <class T> ALL_Point<T>::ALL_Point(const std::vector<T>& data, const T w)
    : ALL_Point<T>(d,data)
{
    weight = w;
}

// unsafe version: no bound checking for data possible
// data needs to be at least of length dimension
template <class T> void ALL_Point<T>::set_coordinates(const T* data)
{
    for (int i = 0; i < dimension; ++i)
        coordinates.at(i) = data[i];
}

template <class T> void ALL_Point<T>::set_coordinates(const std::list<T>& data)
{
    // copy the contents of the list data to the coordinates vector
    coordinates.insert(coordinates.begin(),data.begin(),data.end());
}

template <class T> void ALL_Point<T>::set_coordinates(const std::vector<T>& data)
{
    for (int i = 0; i < dimension; ++i)
        coordinates.at(i) = data.at(i);
}

template <class T> T ALL_Point<T>::get_coordinate(const int idx) const
{
    return coordinates.at(idx);
}

template <class T> void ALL_Point<T>::set_coordinate(const int idx, const T& data)
{
    coordinates.at(idx) = data;
}

template <class T> T ALL_Point<T>::x(int idx) const
{
    return coordinates.at(idx);
}

// calculates euclidian distance between local ALL_Point and another given point
template <class T> double ALL_Point<T>::d(ALL_Point<T> p)
{
    double distance = 0.0;
    int d_p = p.get_dimension();
    if (p != dimension) throw ALL_Point_Dimension_Missmatch_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__
            );
    for (int i = 0; i < dimension; ++i)
    {
        distance += std::pow(coordinates.at(i) - p.get_coordinate(i), 2.0);
    }
    return std::sqrt(distance);
}

// calculates the manhatten distance between local ALL_Point and another given point (one-norm)
template <class T> double ALL_Point<T>::d_1(ALL_Point<T> p)
{
    double distance = 0.0;
    int d_p = p.get_dimension();
    if (p != dimension) throw ALL_Point_Dimension_Missmatch_Exception(
                                        __FILE__,
                                        __func__,
                                        __LINE__
            );
    for (int i = 0; i < dimension; ++i)
    {
        distance += std::abs(coordinates->at(i) - p.get_coordinate(i));
    }
    return distance;
}

template <class T> std::ostream& operator<< (std::ostream& os, const ALL_Point<T>& p)
{
    for (int i = 0; i < p.get_dimension(); ++i)
        os << p.x(i) << " ";
    os << p.get_weight();
    return os;
}
#endif
