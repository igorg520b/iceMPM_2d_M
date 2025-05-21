#include "bezierpath.h"
#include <spdlog/spdlog.h>

void BezierPath::print() const
{
    spdlog::info("Path {}; {}; winding {}; nseg {}",
                 name, (closed ? "Closed" : "Open"), winding, segments.size());
/*
    for (size_t j = 0; j < segments.size(); ++j) {
        const BezierSegment& seg = segments[j];
        spdlog::info("Seg {}: P0 (start) [{},{}], P1 (ctrl1) [{},{}], P2 (ctrl2) [{},{}], P3 (end) [{},{}]",
                     j+1, seg.x0, seg.y0, seg.x1, seg.y1, seg.x2, seg.y2, seg.x3, seg.y3);
    }
    std::cout << "\n";
*/
}

void BezierPath::ComputeWinding()
{
    if (!closed) {
        winding = 0;
        return;
    }

    // Compute signed area using the shoelace formula with sampled points
    float area = 0.0f;
    const float step = 0.1f;
    std::vector<std::pair<float, float>> points;
    float x, y; // Declare x and y here for the entire path scope

    for (const auto& seg : segments) {
        for (float t = 0.0f; t < 1.0f; t += step) {
            evaluateBezier(seg, t, x, y);
            points.emplace_back(x, y);
        }
    }
    // Add the last point (t=1.0 of last segment)
    evaluateBezier(segments.back(), 1.0f, x, y);
    points.emplace_back(x, y);

    // Shoelace formula
    for (size_t i = 0; i < points.size(); ++i) {
        size_t j = (i + 1) % points.size();
        area += (points[i].first * points[j].second) - (points[j].first * points[i].second);
    }
    area /= 2.0f;

    winding = (area < 0.0f) ? 1 : -1;
}

void BezierPath::evaluateBezier(const BezierSegment& seg, float t, float& x, float& y)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;

    x = mt3 * seg.x0 + 3 * mt2 * t * seg.x1 + 3 * mt * t2 * seg.x2 + t3 * seg.x3;
    y = mt3 * seg.y0 + 3 * mt2 * t * seg.y1 + 3 * mt * t2 * seg.y2 + t3 * seg.y3;
}


Eigen::Vector2i BezierPath::evaluateBezierInt(const BezierSegment& seg, float t)
{
    float x,y;
    evaluateBezier(seg, t, x, y);
    Eigen::Vector2i result(static_cast<int>(x), static_cast<int>(y));
    return result;
}


Eigen::Vector2f BezierPath::computeTangent(const BezierSegment& seg, float t)
{
    float t2 = t * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;

    float tx = 3 * mt2 * (seg.x1 - seg.x0) + 6 * mt * t * (seg.x2 - seg.x1) + 3 * t2 * (seg.x3 - seg.x2);
    float ty = 3 * mt2 * (seg.y1 - seg.y0) + 6 * mt * t * (seg.y2 - seg.y1) + 3 * t2 * (seg.y3 - seg.y2);

    Eigen::Vector2f result(tx, ty);
    result.normalize();
    return result;
}




void BezierPath::render(std::vector<Eigen::Vector2f>& normals, std::vector<int>& path_indices,
                        int width, int height, int pathIdx)
{
    spdlog::info("rendering path idx {}", pathIdx);

    // Lambda for index computation
    auto computeIndex = [&](int i, int j) -> size_t {
        return (height - 1 - j)*width + i;
    };

    auto renderPixel = [&](BezierSegment seg, float t, int pathIdx_)  // given segment index and t, render the corresponding pixel
    {
        Eigen::Vector2i pos = evaluateBezierInt(seg, t);
        boundaryPoints.push_back(pos);  // add the rendered pixel to the list even if it's not on the canvas
        if(!(pos.x() >= 0 && pos.x() < width && pos.y() >= 0 && pos.y() < height)) return; // outside of rendering bounds
        Eigen::Vector2f tangent = computeTangent(seg, t);
        Eigen::Vector2f normal(tangent.y(), tangent.x());
        if(winding != 1) normal *= -1;
        if(isMain) normal *= -1;
        size_t index = computeIndex(pos.x(), pos.y());
        normals[index] = normal;
//        path_indices[index] = pathIdx_;
    };

    boundaryPoints.clear();
    Eigen::Vector2i prevPos = evaluateBezierInt(segments[0], 0);

    for (size_t segIdx = 0; segIdx < segments.size(); ++segIdx)
    {
        spdlog::info("rendering path {}; segment {} of {}", pathIdx, segIdx, segments.size());
        const auto& seg = segments[segIdx];

        float step_delta = 1.f / std::max(width, height);
        float t = 0.0f;
        renderPixel(seg, t, pathIdx);

        while (t < 1.0f)
        {
            float tentative_t = t + step_delta;
            if(tentative_t > 1.f) tentative_t = 1.f;
            Eigen::Vector2i tentativePos = evaluateBezierInt(seg, tentative_t);
            Eigen::Vector2i diffPos = tentativePos - prevPos;
            if(diffPos.x() == 0 && diffPos.y() == 0) { t = tentative_t; continue; } // we are still on the same pixel

            bool isAdjacent = (std::abs(diffPos.x()) <= 1 && std::abs(diffPos.y()) <= 1);
            if(!isAdjacent)
            {
                step_delta /= 2;
                if(step_delta < 1e-10f)
                {
                    spdlog::error("path {}; seg {}; t {}; step_delta {}", name, segIdx, t, step_delta);
                    throw std::runtime_error("Failed to render Bézier curve: step size too small");
                }
                continue;
            }

            t = tentative_t;
            renderPixel(seg, t, pathIdx);
            prevPos = tentativePos;
        }
    }
}



//========================================================

void BezierPath::renderFilled(std::vector<int>& path_indices, int width, int height, int pathIdx) const
{
    auto computeIndex = [&](int i, int j) -> size_t {
        return (height - 1 - j)*width + i;
    };

    // Step 1: Compute scanline crossings from boundaryPoints
    std::vector<std::vector<std::pair<float, int>>> scanlines(height); // {x, direction}
    for (size_t i = 1; i < boundaryPoints.size(); ++i) {
        Eigen::Vector2i prev = boundaryPoints[i - 1];
        Eigen::Vector2i curr = boundaryPoints[i];
        int yMin = std::min(prev.y(), curr.y());
        int yMax = std::max(prev.y(), curr.y());
        if (yMin == yMax) continue; // Skip horizontal segments

        float dx = static_cast<float>(curr.x() - prev.x()) / (curr.y() - prev.y());
        for (int y = yMin; y < yMax; ++y) {
            if (y < 0 || y >= height) continue;
            float x = prev.x() + dx * (y + 0.5f - prev.y()); // Midpoint for accuracy
            int dir = (curr.y() > prev.y()) ? 1 : -1;
            scanlines[y].emplace_back(x, dir);
        }
    }

    // Step 2: Fill interior with non-zero winding rule
    int fillValue = 1000 + pathIdx; // All paths use 1000 + pathIdx for interior
    for (int y = 0; y < height; ++y) {
        if (scanlines[y].empty()) continue;
        std::sort(scanlines[y].begin(), scanlines[y].end()); // Sort by x
        int winding = 0;
        float prevX = -1.0f;
        for (const auto& [x, dir] : scanlines[y]) {
            if (winding != 0) {
                int xStart = static_cast<int>(std::ceil(prevX));
                int xEnd = static_cast<int>(std::floor(x));
                for (int xFill = xStart; xFill <= xEnd; ++xFill) {
                    if (xFill >= 0 && xFill < width) {
                        size_t idx = computeIndex(xFill, y);
                        if (path_indices[idx] == -1 || (path_indices[idx] == 1000 && pathIdx != 0)) { // Only fill untouched areas
                            path_indices[idx] = fillValue;
                        }
                    }
                }
            }
            winding += dir;
            prevX = x;
        }
    }

    // Step 3: Overlay boundary points
    for (const auto& pos : boundaryPoints) {
        if (pos.x() >= 0 && pos.x() < width && pos.y() >= 0 && pos.y() < height) {
            size_t idx = computeIndex(pos.x(), pos.y());
            path_indices[idx] = pathIdx; // Boundary gets pathIdx
        }
    }
}

