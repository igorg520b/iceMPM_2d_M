// optimized_poisson_disk_sampling.h
// Based on thinks/poisson-disk-sampling
// Optimized with PagedGrid and Mask support by Antigravity

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include <memory>

#if __cplusplus >= 201402L
#define CONSTEXPR14 constexpr
#else
#define CONSTEXPR14 inline
#endif

namespace thinks {
namespace optimized_pds_internal {

// --- Utils (Copied/Adapted) ---

template <typename ArithT>
CONSTEXPR14 auto squared(const ArithT x) noexcept -> ArithT {
  static_assert(std::is_arithmetic<ArithT>::value, "ArithT must be arithmetic");
  return x * x;
}

template <typename ArithT, std::size_t N>
CONSTEXPR14 auto SquaredMagnitude(const std::array<ArithT, N>& x) noexcept ->
    typename std::array<ArithT, N>::value_type {
  auto m = squared(x[0]);
  for (std::size_t i = 1; i < N; ++i) m += squared(x[i]);
  return m;
}

template <typename VecTraitsT, typename VecT>
CONSTEXPR14 auto SquaredDistance(const VecT& u, const VecT& v) noexcept ->
    typename VecTraitsT::ValueType {
  auto d = squared(VecTraitsT::Get(u, 0) - VecTraitsT::Get(v, 0));
  for (std::size_t i = 1; i < VecTraitsT::kSize; ++i) {
    d += squared(VecTraitsT::Get(u, i) - VecTraitsT::Get(v, i));
  }
  return d;
}

template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
CONSTEXPR14 auto InsideBounds(const VecT& sample,
                              const std::array<FloatT, N>& x_min,
                              const std::array<FloatT, N>& x_max) noexcept -> bool {
  for (std::size_t i = 0; i < VecTraitsT::kSize; ++i) {
    const auto xi = static_cast<FloatT>(VecTraitsT::Get(sample, i));
    if (!(x_min[i] <= xi && xi <= x_max[i])) return false;
  }
  return true;
}

template <typename ArithT, std::size_t N>
CONSTEXPR14 auto ValidBounds(const std::array<ArithT, N>& x_min,
                             const std::array<ArithT, N>& x_max) noexcept -> bool {
  for (std::size_t i = 0; i < N; ++i) {
    if (!(x_max[i] > x_min[i])) return false;
  }
  return true;
}

template <typename T>
void EraseUnordered(std::vector<T>* const v, const std::size_t index) noexcept {
  (*v)[index] = v->back();
  v->pop_back();
}

template <typename IntT, std::size_t N>
CONSTEXPR14 auto Iterate(const std::array<IntT, N>& min_index,
                         const std::array<IntT, N>& max_index,
                         std::array<IntT, N>* const index) noexcept -> bool {
  std::size_t i = 0;
  for (; i < N; ++i) {
    (*index)[i]++;
    if ((*index)[i] <= max_index[i]) break;
    (*index)[i] = min_index[i];
  }
  return i == N ? false : true;
}

// RNG
CONSTEXPR14 auto Hash(const std::uint32_t seed) noexcept -> std::uint32_t {
  auto i = std::uint32_t{(seed ^ 12345391U) * 2654435769U};
  i ^= (i << 6U) ^ (i >> 26U);
  i *= 2654435769U;
  i += (i << 5U) ^ (i >> 12U);
  return i;
}

CONSTEXPR14 auto Rand(std::uint32_t* const seed) noexcept -> std::uint32_t {
  return Hash((*seed)++);
}

template <typename FloatT>
CONSTEXPR14 auto NormRand(std::uint32_t* const seed) noexcept -> FloatT {
  return (1 / static_cast<FloatT>(std::numeric_limits<std::uint32_t>::max())) *
         static_cast<FloatT>(Rand(seed));
}

template <typename FloatT>
CONSTEXPR14 auto RangeRand(const FloatT offset, const FloatT range,
                           std::uint32_t* const seed) noexcept -> FloatT {
  return offset + range * NormRand<FloatT>(seed);
}

template <typename FloatT, std::size_t N>
CONSTEXPR14 auto ArrayRangeRand(const std::array<FloatT, N>& x_min,
                                const std::array<FloatT, N>& x_max,
                                std::uint32_t* const seed) noexcept -> std::array<FloatT, N> {
  std::array<FloatT, N> a = {};
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = RangeRand(x_min[i], x_max[i] - x_min[i], seed);
  }
  return a;
}

template <std::size_t N, typename FloatT>
CONSTEXPR14 auto ArrayRangeRand(const FloatT x_min, const FloatT x_max,
                                std::uint32_t* const seed) noexcept
    -> std::array<FloatT, N> {
  const auto range = x_max - x_min;
  std::array<FloatT, N> a = {};
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = RangeRand(/* offset */ x_min, range, seed);
  }
  return a;
}

template <typename VecT, typename VecTraitsT, typename FloatT, std::size_t N>
CONSTEXPR14 auto VecRangeRand(const std::array<FloatT, N>& x_min,
                              const std::array<FloatT, N>& x_max,
                              std::uint32_t* const seed) noexcept -> VecT {
  VecT v = {};
  const auto a = ArrayRangeRand(x_min, x_max, seed);
  for (std::size_t i = 0; i < N; ++i) {
    VecTraitsT::Set(&v, i, static_cast<typename VecTraitsT::ValueType>(a[i]));
  }
  return v;
}

CONSTEXPR14 auto IndexRand(const std::size_t size,
                           std::uint32_t* const seed) noexcept -> std::size_t {
  constexpr auto kEps = 0.0001F;
  return static_cast<std::size_t>(
      RangeRand(float{0}, static_cast<float>(size) - kEps, seed));
}

// --- PagedGrid Impl ---

template <typename FloatT, std::size_t N>
class PagedGrid {
 public:
  using CellType = std::int32_t;
  using IndexType = std::array<std::int32_t, N>;
  
  // Page size configuration: 32^N cells per page.
  // For 2D: 32x32 = 1024 cells.
  // For 3D: 32x32x32 = 32768 cells.
  static constexpr std::int32_t kPageDim = 32;
  static constexpr std::size_t kCellsPerPage = (N == 2) ? (kPageDim*kPageDim) : (kPageDim*kPageDim*kPageDim);

  explicit PagedGrid(const FloatT sample_radius, const std::array<FloatT, N>& x_min,
                     const std::array<FloatT, N>& x_max) noexcept
      : sample_radius_(sample_radius),
        dx_(GetDx_(sample_radius_)),
        dx_inv_(FloatT{1} / dx_),
        x_min_(x_min),
        size_(GetGridSize_(x_min_, x_max, dx_inv_))
  {
      // Calculate layout of pages
      for(size_t i=0; i<N; ++i) {
          page_grid_size_[i] = (size_[i] + kPageDim - 1) / kPageDim;
      }
      
      std::size_t total_pages = 1;
      for(size_t i=0; i<N; ++i) total_pages *= page_grid_size_[i];
      
      pages_.resize(total_pages, nullptr);
  }

  // Destructor to clean up pages
  ~PagedGrid() {
      for(auto* p : pages_) {
          delete[] p;
      }
  }
  
  // Non-copyable for simplicity
  PagedGrid(const PagedGrid&) = delete;
  PagedGrid& operator=(const PagedGrid&) = delete;

  auto sample_radius() const noexcept -> FloatT { return sample_radius_; }
  auto size() const noexcept -> IndexType { return size_; }

  template <typename FloatT2>
  auto AxisIndex(const std::size_t i, const FloatT2 pos) const noexcept ->
      typename IndexType::value_type {
    return static_cast<typename IndexType::value_type>((static_cast<FloatT>(pos) - x_min_[i]) * dx_inv_);
  }

  template <typename VecTraitsT, typename VecT>
  auto IndexFromSample(const VecT& sample) const noexcept -> IndexType {
    IndexType index = {};
    for (std::size_t i = 0; i < N; ++i) {
      index[i] = AxisIndex(i, VecTraitsT::Get(sample, i));
    }
    return index;
  }

  auto Cell(const IndexType& index) const noexcept -> CellType {
      std::size_t page_idx = GetPageIndex_(index);
      if (pages_[page_idx] == nullptr) return -1;
      return pages_[page_idx][GetCellOffsetInPage_(index)];
  }

  auto Cell(const IndexType& index) noexcept -> CellType& {
      std::size_t page_idx = GetPageIndex_(index);
      if (pages_[page_idx] == nullptr) {
          AllocatePage_(page_idx);
      }
      return pages_[page_idx][GetCellOffsetInPage_(index)];
  }

 private:
  FloatT sample_radius_;
  FloatT dx_;
  FloatT dx_inv_;
  std::array<FloatT, N> x_min_;
  IndexType size_;
  
  IndexType page_grid_size_; // Dimensions in pages
  std::vector<CellType*> pages_; // Vector of pointers to pages

  void AllocatePage_(std::size_t page_idx) {
      CellType* new_page = new CellType[kCellsPerPage];
      std::fill(new_page, new_page + kCellsPerPage, -1);
      pages_[page_idx] = new_page;
  }

  std::size_t GetPageIndex_(const IndexType& index) const noexcept {
      // Basic linear indexing for pages
      // page_idx = (idx[0]/kPageDim) + (idx[1]/kPageDim) * page_grid_size_[0] ...
      // Assuming N=2 for speed optimization details, but generalized loop works.
      std::size_t p_idx = 0;
      std::size_t stride = 1;
      for (size_t i = 0; i < N; ++i) {
          p_idx += (static_cast<std::size_t>(index[i]) / kPageDim) * stride;
          stride *= static_cast<std::size_t>(page_grid_size_[i]);
      }
      return p_idx;
  }

  std::size_t GetCellOffsetInPage_(const IndexType& index) const noexcept {
      // offset = (idx[0]%kPageDim) + (idx[1]%kPageDim) * kPageDim ...
      std::size_t offset = 0;
      std::size_t stride = 1;
      for (size_t i = 0; i < N; ++i) {
          offset += (static_cast<std::size_t>(index[i]) % kPageDim) * stride;
          stride *= kPageDim; 
      }
      return offset;
  }

  static auto GetDx_(const FloatT sample_radius) noexcept -> FloatT {
    constexpr auto kEps = static_cast<FloatT>(0.001);
    constexpr auto kScale = FloatT{1} - kEps;
    return kScale * sample_radius / std::sqrt(static_cast<FloatT>(N));
  }

  static auto GetGridSize_(const std::array<FloatT, N>& x_min,
                           const std::array<FloatT, N>& x_max,
                           const FloatT dx_inv) noexcept -> IndexType {
    IndexType grid_size = {};
    for (std::size_t i = 0; i < N; ++i) {
      grid_size[i] = static_cast<typename IndexType::value_type>(
          std::ceil((x_max[i] - x_min[i]) * dx_inv));
    }
    return grid_size;
  }
};

template <typename FloatT, std::size_t N>
auto MakePagedGrid(const FloatT sample_radius, const std::array<FloatT, N>& x_min,
                   const std::array<FloatT, N>& x_max) noexcept -> PagedGrid<FloatT, N> {
  return PagedGrid<FloatT, N>(sample_radius, x_min, x_max);
}

template <typename VecT>
struct ActiveSample {
  VecT position;
  std::size_t active_index;
  std::uint32_t sample_index;
};

template <typename VecT>
auto RandActiveSample(const std::vector<std::uint32_t>& active_indices,
                      const std::vector<VecT>& samples,
                      std::uint32_t* const seed) noexcept
    -> ActiveSample<VecT> {
  ActiveSample<VecT> active_sample = {};
  active_sample.active_index = IndexRand(active_indices.size(), seed);
  active_sample.sample_index = active_indices[active_sample.active_index];
  active_sample.position = samples[active_sample.sample_index];
  return active_sample;
}

template <typename VecTraitsT, typename VecT, typename FloatT>
CONSTEXPR14 auto RandAnnulusSample(const VecT& center, const FloatT radius,
                                   std::uint32_t* const seed) noexcept -> VecT {
  VecT p = {};
  for (;;) {
    const auto offset =
        ArrayRangeRand<VecTraitsT::kSize>(FloatT{-2}, FloatT{2}, seed);
    const auto r2 = SquaredMagnitude(offset);
    if (FloatT{1} < r2 && r2 <= FloatT{4}) {
      for (std::size_t i = 0; i < VecTraitsT::kSize; ++i) {
        VecTraitsT::Set(&p, i,
                        static_cast<typename VecTraitsT::ValueType>(
                            static_cast<FloatT>(VecTraitsT::Get(center, i)) +
                            radius * offset[i]));
      }
      break;
    }
  }
  return p;
}

template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
void AddSample(const VecT& sample, std::vector<VecT>* const samples,
               std::vector<std::uint32_t>* const active_indices,
               PagedGrid<FloatT, N>* const grid) noexcept {
  const auto sample_index = samples->size();
  samples->push_back(sample);
  active_indices->push_back(static_cast<std::uint32_t>(sample_index));
  const auto grid_index = grid->template IndexFromSample<VecTraitsT>(sample);
  grid->Cell(grid_index) = static_cast<std::int32_t>(sample_index);
}

template <typename IndexT>
struct GridIndexRange {
  IndexT min_index;
  IndexT max_index;
};

template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
auto GridNeighborhood(const VecT& sample, const PagedGrid<FloatT, N>& grid) noexcept
    -> GridIndexRange<typename PagedGrid<FloatT, N>::IndexType> {
  auto min_index = typename PagedGrid<FloatT, N>::IndexType{};
  auto max_index = typename PagedGrid<FloatT, N>::IndexType{};
  const auto grid_size = grid.size();
  const auto radius = grid.sample_radius();
  for (auto i = std::size_t{0}; i < N; ++i) {
    const auto xi_min = 0;
    const auto xi_max = grid_size[i] - 1;
    const auto xi = static_cast<FloatT>(VecTraitsT::Get(sample, i));
    const auto xi_sub = grid.AxisIndex(i, xi - radius);
    const auto xi_add = grid.AxisIndex(i, xi + radius);
    min_index[i] = (xi_sub < xi_min) ? xi_min : (xi_sub > xi_max ? xi_max : xi_sub); 
    max_index[i] = (xi_add < xi_min) ? xi_min : (xi_add > xi_max ? xi_max : xi_add);
  }
  return {min_index, max_index};
}

template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
auto ExistingSampleWithinRadius(
    const VecT& sample, const std::uint32_t active_sample_index,
    const std::vector<VecT>& samples, const PagedGrid<FloatT, N>& grid,
    const typename PagedGrid<FloatT, N>::IndexType& min_index,
    const typename PagedGrid<FloatT, N>::IndexType& max_index) noexcept -> bool {
  auto index = min_index;
  const auto r_squared = squared(grid.sample_radius());
  do {
    const auto cell_index = grid.Cell(index);
    if (cell_index >= 0 &&
        static_cast<std::uint32_t>(cell_index) != active_sample_index) {
      const auto cell_sample = samples[static_cast<std::uint32_t>(cell_index)];
      const auto d =
          static_cast<FloatT>(SquaredDistance<VecTraitsT>(sample, cell_sample));
      if (d < r_squared) {
        return true;
      }
    }
  } while (Iterate(min_index, max_index, &index));

  return false;
}

} // namespace optimized_pds_internal

// --- Public Entry Point with Mask ---

// Mask function returns TRUE if the point is valid (inside ice), FALSE to reject.
template <typename FloatT, std::size_t N, typename MaskFuncT, typename VecT = std::array<FloatT, N>,
          typename VecTraitsT = thinks::VecTraits<VecT>>
auto PoissonDiskSampling(const FloatT radius,
                         const std::array<FloatT, N>& x_min,
                         const std::array<FloatT, N>& x_max,
                         MaskFuncT mask, // <-- Generic Mask Argument
                         const std::uint32_t max_sample_attempts = 30,
                         const std::uint32_t seed = 0) noexcept
    -> std::vector<VecT> {
  namespace pds = optimized_pds_internal;

  if (!(radius > FloatT{0}) || !(max_sample_attempts > 0) ||
      !pds::ValidBounds(x_min, x_max)) {
    return std::vector<VecT>{};
  }

  // Use PagedGrid for memory optimization
  auto grid = pds::MakePagedGrid(radius, x_min, x_max);

  auto samples = std::vector<VecT>{};
  auto active_indices = std::vector<std::uint32_t>{};
  auto local_seed = seed;

  // Add first sample: Must be inside MASK
  // Naive strategy: Try random points until one fits mask.
  // Safety break after e.g. 10000 tries to avoid infinite loop if mask is empty.
  bool first_found = false;
  for(int i=0; i<10000; ++i) {
      VecT p = pds::VecRangeRand<VecT, VecTraitsT>(x_min, x_max, &local_seed);
      if (mask(p)) {
          pds::AddSample<VecTraitsT>(p, &samples, &active_indices, &grid);
          first_found = true;
          break;
      }
  }
  
  if (!first_found) return samples; // Empty result if no start point found

  // MAIN LOOP with RESTART support for Disconnected Components
  while (true) {
      
      // 1. Bridson's Algorithm (Fill connected component)
      while (!active_indices.empty()) {
        const auto active_sample =
            pds::RandActiveSample(active_indices, samples, &local_seed);

        auto attempt_count = std::uint32_t{0};
        while (attempt_count < max_sample_attempts) {
          const auto cand_sample = pds::RandAnnulusSample<VecTraitsT>(
              active_sample.position, grid.sample_radius(), &local_seed);

          // 1. Check bounds
          if (pds::InsideBounds<VecTraitsT>(cand_sample, x_min, x_max)) {
              // 2. Check Mask (Optimization: Don't check grid if mask fails)
              if (mask(cand_sample)) {
                    // 3. Check Grid Neighbors
                    const auto grid_neighbors =
                        pds::GridNeighborhood<VecTraitsT>(cand_sample, grid);
                    const auto existing_sample =
                        pds::ExistingSampleWithinRadius<VecTraitsT>(
                            cand_sample, active_sample.sample_index, samples, grid,
                            grid_neighbors.min_index, grid_neighbors.max_index);
                    if (!existing_sample) {
                      pds::AddSample<VecTraitsT>(cand_sample, &samples, &active_indices,
                                                 &grid);
                      break;
                    }
              }
          }

          ++attempt_count;
        }

        if (attempt_count == max_sample_attempts) {
          pds::EraseUnordered(&active_indices, active_sample.active_index);
        }
      }

      // 2. Restart Phase: Try to find a new seed in a potentially disconnected region.
      // We perform global dart throwing. If we find a valid point not covered by existing samples,
      // we add it and resume Bridson's.
      bool found_restart_seed = false;
      constexpr int kRestartAttempts = 2000; // Heuristic attempts to find a hole
      
      for(int i=0; i<kRestartAttempts; ++i) {
          VecT p = pds::VecRangeRand<VecT, VecTraitsT>(x_min, x_max, &local_seed);
          
          if (mask(p)) {
             // Check if 'p' is too close to ANY existing sample.
             // We can check this by querying the grid efficiently.
             const auto grid_neighbors =
                    pds::GridNeighborhood<VecTraitsT>(p, grid);
             
             // Note: We use a dummy index for 'active_sample_index' since 'p' is new.
             const auto existing_sample =
                 pds::ExistingSampleWithinRadius<VecTraitsT>(
                     p, std::numeric_limits<std::uint32_t>::max(), samples, grid,
                     grid_neighbors.min_index, grid_neighbors.max_index);
             
             if (!existing_sample) {
                 // Found a new seed!
                 pds::AddSample<VecTraitsT>(p, &samples, &active_indices, &grid);
                 found_restart_seed = true;
                 break; // Go back to filling
             }
          }
      }
      
      if (!found_restart_seed) {
          break; // We tried hard and found nothing. Assume domain is full.
      }
      // Else: loop back to Bridson's with the newly added seed in 'active_indices'
  }

  return samples;
}

} // namespace thinks
