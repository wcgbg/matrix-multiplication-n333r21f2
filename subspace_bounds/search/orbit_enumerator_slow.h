#pragma once

// Orbit enumerator: classify all subspaces of (𝔽_P^NA)* under the SymmetryGroup
// action.
//
// Reference/slow version: for every candidate constraint set it sweeps all
// (query⁻¹, store) pairs — the product set Query⁻¹ · Store, which covers the
// whole group G (no square-root trick) — and keeps the set only if it is the
// lexicographic minimum of its orbit. This is the SAME covering set
// OrbitMap::Get uses (it connects q ≡ query_elem⁻¹·store_elem·c), so the
// enumerator and the prover/verifier agree on orbits under a single coverage
// requirement. Coverage is all this needs; the two sides are not assumed to be
// subgroups (see core/symmetry.h). This is the correctness oracle the fast
// OrbitEnumerator is tested against, so it must emit exactly the same canonical
// representatives.
//
// Each action is query.ApplyInverse(query_elem, store.Apply(store_elem, v))
// on a GFVec<Problem::kP, Problem::kNA> row.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "boost/unordered/unordered_flat_set.hpp"
#include "ng-log/logging.h"

#include "core/bit_vec.h"
#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/gf_vec.h"
#include "core/symmetry.h"
#include "core/tensor.h"
#include "core/tensor_utils.h"

template <class Problem> class OrbitEnumeratorSlow {
public:
  static constexpr int kNA = Problem::kNA;
  static constexpr int kNB = Problem::kNB;
  static constexpr int kNC = Problem::kNC;
  static constexpr int kP = Problem::kP;
  using Vec = GFVec<kP, kNA>;
  using SymmetryGroup = typename Problem::SymmetryGroup;
  using QueryElem = typename SymmetryGroup::QuerySet::Elem;
  using StoreElem = typename SymmetryGroup::StoreSet::Elem;

  // The DFS sweeps all q^kNA = P^(kM·kNA) dual vectors, so the slow
  // enumerator is only practical for tiny kNA (it is the reference impl for
  // unit tests). The bound keeps the iteration count fit in a uint64_t with
  // headroom.
  static_assert(IntPow(kP, kNA) <= (uint64_t{1} << 30),
                "OrbitEnumeratorSlow: too many candidates "
                "(P^kNA > 2^30)");

  // The group is borrowed; the caller owns it and keeps it alive.
  explicit OrbitEnumeratorSlow(const SymmetryGroup *group) : group_(group) {
    CHECK_NOTNULL(group_);
  }

  // Enumerate one canonical representative per orbit, partitioned by subspace
  // dimension (descending in the output, as the DP driver consumes them).
  pb::Certificate Search(bool fill_verbose_fields = false) const {
    Set minimal;
    Constraints<kP, kNA> constraints;
    CHECK(minimal.insert(constraints).second); // the zero subspace (dim 0)
    SearchRec(constraints, &minimal);

    const Tensor<kP, kNA, kNB, kNC> tensor = Problem::MakeTensor();
    std::vector<std::vector<Constraints<kP, kNA>>> dim_to_constraints(kNA + 1);
    for (const auto &r : minimal) {
      dim_to_constraints.at(r.size()).push_back(r);
    }

    pb::Certificate certificate;
    certificate.set_problem_name(Problem::Name());
    certificate.set_na(kNA);
    certificate.set_nb(kNB);
    certificate.set_nc(kNC);
    certificate.set_characteristic(kP);
    certificate.set_extension_degree(1);
    for (int dim = kNA; dim >= 0; --dim) {
      LOG(INFO) << "dim=" << dim << " count=" << dim_to_constraints[dim].size();
      std::sort(dim_to_constraints[dim].begin(), dim_to_constraints[dim].end());
      for (const Constraints<kP, kNA> &r : dim_to_constraints[dim]) {
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
  using Set =
      boost::unordered_flat_set<Constraints<kP, kNA>, ConstraintsHash<kP, kNA>>;

  // DFS over strictly-increasing sequences of dual vectors. Each increasing
  // sequence is a distinct candidate set; Visit decides if it is the canonical
  // orbit representative (and prunes its subtree if not).
  void SearchRec(Constraints<kP, kNA> constraints, Set *minimal) const {
    if (constraints.size() == static_cast<std::size_t>(kNA)) {
      return; // already spans (𝔽_P^kNA)*; nothing independent can be appended
    }
    static_assert(kNA < 64);
    // Encode each candidate by an integer in [0, q^kNA) = [0, P^(kM·kNA));
    // the ordering matches the GFVec lex (and the F_2 bit-packed integer
    // order) so "strictly increasing" amounts to integer comparison.
    constexpr uint64_t kMax = IntPow(kP, kNA);
    uint64_t last_int = 0;
    if (!constraints.empty()) {
      last_int = EncodeGFVec<kP, kNA>(constraints.back());
    }
    for (uint64_t data = last_int + 1; data < kMax; ++data) {
      Vec candidate = DecodeGFVec<kP, kNA>(data);
      if (candidate.LeadingNonzero() != GF<kP>::One()) {
        continue;
      }
      constraints.push_back(candidate);
      if (Visit(constraints, minimal)) {
        SearchRec(constraints, minimal);
      }
      constraints.pop_back();
    }
  }

  // Apply the group element (query⁻¹·store) to every functional, then reduce
  // to (column-reversed) RREF and drop the zero rows the reversed elimination
  // parks at the front. The result has size == rank, matching OrbitMap's keys.
  // query⁻¹·store so the swept set is Query⁻¹ · Store, the
  // same covering set OrbitMap::Get uses — see the file header.
  Constraints<kP, kNA> Transform(QueryElem query, StoreElem store,
                                 const Constraints<kP, kNA> &in) const {
    Constraints<kP, kNA> out;
    out.reserve(in.size());
    for (const Vec v : in) {
      out.push_back(
          group_->query.ApplyInverse(query, group_->store.Apply(store, v)));
    }
    const int rank = GaussJordanRREF<kP, kNA>(&out);
    out.erase(out.begin(), out.end() - rank);
    return out;
  }

  // True iff `constraints` is the lexicographic minimum of its orbit (and is
  // linearly independent). Identity ∈ G, so a survivor equals its own RREF —
  // hence the stored representatives are canonical. A dependent candidate is
  // caught at the identity element, where the RREF has fewer rows.
  bool Visit(const Constraints<kP, kNA> &constraints, Set *minimal) const {
    const int query_size = group_->query.Size();
    const int store_size = group_->store.Size();
    for (int i = 0; i < query_size; ++i) {
      const QueryElem query = group_->query.At(i);
      for (int j = 0; j < store_size; ++j) {
        const StoreElem store = group_->store.At(j);
        const Constraints<kP, kNA> image = Transform(query, store, constraints);
        if (image.size() != constraints.size()) {
          return false; // linearly dependent candidate
        }
        if (image < constraints) {
          return false; // not the orbit minimum
        }
      }
    }
    CHECK(minimal->insert(constraints).second);
    return true;
  }

  const SymmetryGroup *group_ = nullptr;
};
