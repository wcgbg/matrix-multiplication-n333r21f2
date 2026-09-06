#pragma once

// Orbit enumerator (fast): classify all subspaces of (𝔽_q^NA)* under the
// SymmetryGroup action, emitting one canonical representative per orbit.
//
// This is the production counterpart of core/orbit_enumerator_slow.h. It emits
// EXACTLY the same canonical representatives (the lexicographic minimum of each
// orbit, in column-reversed RREF) — the slow version is the correctness oracle
// and both are checked against the same golden certificates — but reaches them
// far faster via two ideas ported from rank_search/constraints_enumerator.h:
//
//   1. Layered incremental construction. Orbit reps of dimension k are built by
//      appending one new highest-pivot row to a kept rep of dimension k−1. The
//      child is already in column-reversed RREF by construction (the appended
//      row's pivot sits strictly above every parent pivot, and its lower
//      coordinates avoid the parent pivot columns), so no candidate is ever a
//      dependent or non-canonical form. Candidates are generated in global
//      lexicographic order, so the FIRST member of each orbit we meet is its
//      lex-min — matching the slow enumerator without an explicit lex check.
//
//   2. Meet-in-the-middle dedup (the square-root trick), identical in structure
//      to core/orbit_map.h. For each kept rep c we store every Store-image
//      RREF(store.Apply(store_elem, c)) in a hash set; a new candidate q is a
//      duplicate iff some Query-image RREF(query.Apply(query_elem, q)) is
//      already stored. A hit means query.Apply(query_elem, q) ≡
//      store.Apply(store_elem, c), i.e. q ≡ query_elem⁻¹·store_elem·c lies in
//      c's orbit. Coverage of Query⁻¹·Store over G (see core/symmetry.h) makes
//      this find every orbit member.
//
// Determinism: candidates are visited SEQUENTIALLY (so first-seen = lex-min);
// the only parallelism is inside Visit, over the group elements of a single
// candidate.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "boost/unordered/unordered_flat_set.hpp"
#include "ng-log/logging.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/gf_vec.h"
#include "core/symmetry.h"
#include "core/tensor.h"
#include "core/tensor_utils.h"

template <class Problem> class OrbitEnumerator {
public:
  static constexpr int kNA = Problem::kNA;
  static constexpr int kNB = Problem::kNB;
  static constexpr int kNC = Problem::kNC;
  static constexpr int kP = Problem::kP;
  using Vec = GFVec<kP, kNA>;
  using SymmetryGroup = typename Problem::SymmetryGroup;
  using QueryElem = typename SymmetryGroup::QuerySet::Elem;
  using StoreElem = typename SymmetryGroup::StoreSet::Elem;

  // The group is borrowed; the caller owns it and keeps it alive.
  explicit OrbitEnumerator(const SymmetryGroup *group) : group_(group) {
    CHECK_NOTNULL(group_);
  }

  // Enumerate one canonical representative per orbit, partitioned by subspace
  // dimension (descending in the output, as the DP driver consumes them).
  pb::Certificate Search(bool fill_verbose_fields = false) const {
    std::vector<std::vector<Constraints<kP, kNA>>> dim_to_reps(kNA + 1);
    dim_to_reps[0].push_back(Constraints<kP, kNA>()); // zero subspace

    for (int dim = 1; dim <= kNA; ++dim) {
      ShardedSet visited; // Store-images of this layer's kept reps only.
      std::vector<Constraints<kP, kNA>> &layer = dim_to_reps[dim];
      // Parents arrive in lexicographic order (see file header), and children
      // of a smaller parent are all lexicographically smaller, so processing
      // them in order keeps generation == global lex order.
      for (const Constraints<kP, kNA> &parent : dim_to_reps[dim - 1]) {
        ExpandNextLayer(parent, &visited, &layer);
      }
      LOG(INFO) << "dim=" << dim << " count=" << layer.size();
    }

    const Tensor<kP, kNA, kNB, kNC> tensor = Problem::MakeTensor();
    pb::Certificate certificate;
    certificate.set_problem_name(Problem::Name());
    certificate.set_na(kNA);
    certificate.set_nb(kNB);
    certificate.set_nc(kNC);
    certificate.set_characteristic(kP);
    certificate.set_extension_degree(1);
    for (int dim = kNA; dim >= 0; --dim) {
      // Already lex-ordered, but sort to match the slow enumerator's output
      // stage exactly (and to be robust to any future generation reordering).
      std::sort(dim_to_reps[dim].begin(), dim_to_reps[dim].end());
      for (const Constraints<kP, kNA> &r : dim_to_reps[dim]) {
        const int index = certificate.constrained_tensors_size();
        pb::ConstrainedTensor *rt = certificate.add_constrained_tensors();
        rt->set_index(index);
        rt->set_constraints(ConstraintsToBytes<kP, kNA>(r));
        if (fill_verbose_fields) {
          rt->set_constraints_text(ConstraintsToString<kP, kNA>(r));
          rt->set_tensor_text(TensorToSparseString<kP, kNA, kNB, kNC>(
              ApplyConstraintsToTensor<kP, kNA, kNB, kNC>(r, tensor)));
        }
      }
    }
    LOG(INFO) << "total_count=" << certificate.constrained_tensors_size();
    return certificate;
  }

private:
  // Sharded hash set of Store-images. Reads (ContainsUnsafe) and writes
  // (Insert) never overlap — Visit probes the whole set before any insert for
  // the same candidate, and candidates are processed sequentially — so the
  // lock-free read is safe.
  class ShardedSet {
  public:
    void Insert(const Constraints<kP, kNA> &r) {
      const std::size_t shard = Hash(r);
      std::lock_guard<std::mutex> lock(mutexes_[shard]);
      sets_[shard].insert(r);
    }

    bool ContainsUnsafe(const Constraints<kP, kNA> &r) const {
      return sets_[Hash(r)].contains(r);
    }

  private:
    static constexpr int kNumShards = 997;
    static std::size_t Hash(const Constraints<kP, kNA> &r) {
      return ConstraintsHash<kP, kNA>{}(r) % kNumShards;
    }
    std::array<boost::unordered_flat_set<Constraints<kP, kNA>,
                                         ConstraintsHash<kP, kNA>>,
               kNumShards>
        sets_;
    std::array<std::mutex, kNumShards> mutexes_;
  };

  // Generate every child of `parent` (one larger dimension) that extends the
  // parent's column-reversed RREF, keeping those that are new orbit reps.
  void ExpandNextLayer(const Constraints<kP, kNA> &parent,
                       ShardedSet *visited,
                       std::vector<Constraints<kP, kNA>> *layer) const {
    // Mark the parent's pivot columns; the highest is the bar the new row's
    // pivot must clear.
    std::array<bool, kNA> is_pivot{};
    int c_max = -1;
    for (const Vec &row : parent) {
      const int piv = row.LeadingNonzeroIdx();
      DCHECK_GE(piv, 0);
      is_pivot[piv] = true;
      c_max = std::max(c_max, piv);
    }

    constexpr uint64_t kQ = kP;
    using Field = GF<kP>;

    Constraints<kP, kNA> child = parent;
    // New pivot column p strictly above every parent pivot.
    for (int p = c_max + 1; p < kNA; ++p) {
      // Free coordinates: columns below p that are not parent pivots. Ordered
      // ascending so free_cols[j] carries the q^j digit of the counter; since
      // the highest free column is the most significant GFVec coordinate, the
      // counter then runs through rows in ascending lexicographic order.
      std::array<int, kNA> free_cols{};
      int f = 0;
      for (int c = 0; c < p; ++c) {
        if (!is_pivot[c]) {
          free_cols[f++] = c;
        }
      }
      const uint64_t end = IntPow(kP, f); // q^f
      for (uint64_t code = 0; code < end; ++code) {
        Vec row{};
        row.Set(p, Field::One());
        uint64_t rest = code;
        for (int j = 0; j < f; ++j) {
          row.Set(free_cols[j], Field{static_cast<uint8_t>(rest % kQ)});
          rest /= kQ;
        }
        child.push_back(row); // already canonical RREF — no re-elimination
        if (Visit(child, visited)) {
          layer->push_back(child);
        }
        child.pop_back();
      }
    }
  }

  // Apply a group element to every row, then reduce to column-reversed RREF and
  // drop the zero rows the reversed elimination parks at the front, so the
  // result is a full-rank canonical key.
  template <class Apply>
  Constraints<kP, kNA> Image(Apply apply,
                                 const Constraints<kP, kNA> &in) const {
    Constraints<kP, kNA> out;
    out.reserve(in.size());
    for (const Vec v : in) {
      out.push_back(apply(v));
    }
    const int rank = GaussJordanRREF<kP, kNA>(&out);
    out.erase(out.begin(), out.end() - rank);
    return out;
  }

  // True iff `candidate` is a new orbit (its lex-min rep, given the generation
  // order). Probes the Query-images against the stored Store-images (the
  // square-root trick, same hit equation as OrbitMap::Get); if none match, the
  // candidate opens a new orbit and we register all its Store-images.
  bool Visit(const Constraints<kP, kNA> &candidate,
             ShardedSet *visited) const {
    const int query_size = group_->query.Size();
    std::atomic<bool> duplicate = false;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, query_size),
        [&](const tbb::blocked_range<int> &range) {
          for (int i = range.begin(); i != range.end(); ++i) {
            if (duplicate.load(std::memory_order_relaxed)) {
              return;
            }
            const QueryElem query_elem = group_->query.At(i);
            const Constraints<kP, kNA> image = Image(
                [&](const Vec v) { return group_->query.Apply(query_elem, v); },
                candidate);
            if (visited->ContainsUnsafe(image)) {
              duplicate.store(true, std::memory_order_relaxed);
              return;
            }
          }
        });
    if (duplicate.load(std::memory_order_relaxed)) {
      return false;
    }

    const int store_size = group_->store.Size();
    tbb::parallel_for(
        tbb::blocked_range<int>(0, store_size),
        [&](const tbb::blocked_range<int> &range) {
          for (int j = range.begin(); j != range.end(); ++j) {
            const StoreElem store_elem = group_->store.At(j);
            visited->Insert(Image(
                [&](const Vec v) { return group_->store.Apply(store_elem, v); },
                candidate));
          }
        });
    return true;
  }

  const SymmetryGroup *group_ = nullptr;
};
