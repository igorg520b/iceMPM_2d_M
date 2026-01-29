#ifndef HOSTSIDESOA_H
#define HOSTSIDESOA_H

#include <utility>
#include <algorithm>

#include <Eigen/Core>
#include <spdlog/spdlog.h>

#include "parameters_sim.h"
#include "proxypoint2d.h"



class SOAIterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = ProxyPoint;
    using difference_type = int;
    using pointer = ProxyPoint*;
    using reference = ProxyPoint; 

    // ProxyPoint m_point; // removed

    SOAIterator(unsigned pos, double *soa_data, unsigned pitch);
    SOAIterator(const SOAIterator& other);
    SOAIterator& operator=(const SOAIterator& other);
    SOAIterator() {};


    bool operator<(const SOAIterator& t) const {return (pos<t.pos && soa==t.soa);}
    bool operator==(const SOAIterator& t)const{return (pos == t.pos && soa==t.soa);}
    bool operator!=(const SOAIterator& t)const{return (pos != t.pos || soa!=t.soa);}

    SOAIterator& operator+=(difference_type n) { pos+=n; return (*this); }
    SOAIterator& operator-=(difference_type n) { pos-=n; return (*this); }
    SOAIterator& operator++() { ++pos; return (*this); }
    SOAIterator& operator--() { --pos; return (*this); }
    SOAIterator operator+(const difference_type& m) const {SOAIterator r=*this;r.pos+=m;return r;}
    SOAIterator operator-(const difference_type& m) const {SOAIterator r=*this;r.pos-=m;return r;}
    difference_type operator-(const SOAIterator& rawIterator) const {return pos-rawIterator.pos;}

    ProxyPoint operator*() const {
        return ProxyPoint(pos, soa, pitch);
    }
    
    struct ArrowProxy {
        ProxyPoint p;
        ProxyPoint* operator->() { return &p; }
    };
    ArrowProxy operator->() const { return ArrowProxy{operator*()}; }

    // Members
    unsigned pos;
    double *soa;
    unsigned pitch;

    friend void iter_swap(const SOAIterator& a, const SOAIterator& b) {
        // Explicitly construct references (avoiding Copy Ctor which would make them Values)
        ProxyPoint pa(a.pos, a.soa, a.pitch);
        ProxyPoint pb(b.pos, b.soa, b.pitch);
        
        using std::swap;
        swap(pa, pb);
    }
};


// the class manages the Structure-of-Arrays (SOA) buffer
class HostSideSOA
{
public:
    HostSideSOA() = default;
    ~HostSideSOA();

    double *host_buffer = nullptr; // buffer in page-locked memory for transferring the data between device and host
    unsigned capacity = 0;  // max number of points that the host-side buffer can hold
    unsigned size = 0;      // the number of points, including "disabled" ones, in the host buffer (may fluctuate)

    SOAIterator begin(){return SOAIterator(0, host_buffer, capacity);}
    SOAIterator end(){return SOAIterator(size, host_buffer, capacity);}

    void Allocate(int pts_capacity);
    void RemoveDisabledAndSort(int GridY);

    double* getPointerToLine(int idxLine) {return host_buffer + capacity*idxLine;}

    std::pair<Eigen::Vector2d, Eigen::Vector2d> getBlockDimensions();
    void convertToIntegerCellFormat(double h);

    // debugging / testing

};

#endif // HOSTSIDESOA_H
