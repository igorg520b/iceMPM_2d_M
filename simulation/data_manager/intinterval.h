#ifndef INTINTERVAL_H
#define INTINTERVAL_H


#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>

class IntInterval {
public:
    int start;
    int end;

    IntInterval(int s, int e) : start(s), end(e) {
        if (e < s) {
            throw std::invalid_argument("End must be >= start");
        }
    }

    // Length of interval
    int size() const {
        return end - start;
    }

    // Intersection with another interval
    IntInterval intersect(const IntInterval& other) const {
        int new_start = std::max(start, other.start);
        int new_end = std::min(end, other.end);
        if (new_start >= new_end) {
            // Return an empty interval [0,0)
            return IntInterval(0, 0);
        }
        return IntInterval(new_start, new_end);
    }

    // Offset of this interval within another (other must fully contain this)
    int offset_within(const IntInterval& outer) const {
        if (start < outer.start || end > outer.end) {
            throw std::out_of_range("This interval is not within the outer interval");
        }
        return start - outer.start;
    }

    // For debugging / logging
    friend std::ostream& operator<<(std::ostream& os, const IntInterval& iv) {
        os << "[" << iv.start << ", " << iv.end << ")";
        return os;
    }
};



#endif // INTINTERVAL_H
