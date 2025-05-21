#ifndef BEZIERPATH_H
#define BEZIERPATH_H

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <Eigen/Core>


// Structure to represent a closed path (sequence of Bézier segments)
struct BezierPath
{
    struct BezierSegment // Structure to represent a single cubic Bézier segment (P0, P1, P2, P3)
    {
        float x0, y0; // Start point (P0)
        float x1, y1; // Control point 1 (P1)
        float x2, y2; // Control point 2 (P2)
        float x3, y3; // End point (P3)

        BezierSegment(float x0_, float y0_, float x1_, float y1_, float x2_, float y2_, float x3_, float y3_)
            : x0(x0_), y0(y0_), x1(x1_), y1(y1_), x2(x2_), y2(y2_), x3(x3_), y3(y3_) {}
    };


    std::vector<BezierSegment> segments;
    bool closed;
    int winding; // 1 for counterclockwise, -1 for clockwise
    std::pair<int, int> seedPoint; // Seed point coordinates (x, y)
    std::string name; // Add name field to store the shape's ID or title
    bool isMain;
    std::vector<Eigen::Vector2i> boundaryPoints;

    BezierPath() : isMain(false), closed(false), winding(0), seedPoint(-1, -1) {}

    void print() const;
    void ComputeWinding();
    void render(std::vector<Eigen::Vector2f>& normals, std::vector<int>& path_indices,
                int width, int height, int pathIdx);
    void renderFilled(std::vector<int>& path_indices, int width, int height, int pathIdx) const;


private:
    static void evaluateBezier(const BezierSegment& seg, float t, float& x, float& y);
    static Eigen::Vector2i evaluateBezierInt(const BezierSegment& seg, float t);
    static Eigen::Vector2f computeTangent(const BezierSegment& seg, float t);
};

#endif // BEZIERPATH_H
