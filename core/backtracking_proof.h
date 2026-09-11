#pragma once

// Compressed on-disk trace of the Substitution-with-Backtracking proof
// (Algorithm 2). Each leaf records the (query_elem, store_elem) witness
// returned by OrbitMap::Get as two uint32 values. These encode matrices and
// packed transformations, not positions in the group enumeration.
//
// One record per DFS leaf at which the prover reached the target bound. The
// records appear in DFS pre-order; the verifier replays the same deterministic
// DFS over the base subspace's minimal constraints and consumes them in order
// (see subspace_bounds/verifier/backtracking_verifier.h).
//
// All of a certificate's per-orbit traces are stored together in a single
// BacktrackingProofArchive file, `<cert_without_extension>.btp`: a small header
// (magic, version, count) followed by `count` length-prefixed, individually
// gzip-compressed proof blobs, one per orbit in dense-index order. A
// zero-length blob means the orbit has no backtracking proof. The file is NOT
// gzipped as a whole — each proof is compressed independently so the archive is
// held (and loaded) in memory in compressed form, decompressing one proof at a
// time.

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "ng-log/logging.h"

// Derive the single per-certificate archive path from the certificate's path,
// e.g. "foo/cert.pb.txt" -> "foo/cert.btp" (and ".pb" -> ".btp").
std::string GetBacktrackingProofArchivePath(const std::string &proto_path);

struct BacktrackingProof {
  void Save(const std::string &path) const;
  static BacktrackingProof Load(const std::string &path);

  // Raw (un-gzipped) body codec. The serialized body is the self-describing
  // `uint64 n` followed, when n > 0, by the four parallel arrays (an empty
  // proof is just `n == 0`).
  void WriteBody(std::ostream &out) const;
  static BacktrackingProof ReadBody(std::istream &in);

  // gzip of WriteBody() as an in-memory string, and its inverse. Used by
  // BacktrackingProofArchive to keep one compressed blob per orbit in memory
  // and (de)compress a single proof on the fly. Empty round-trips through
  // empty: an empty proof compresses to "" and "" decompresses to a default
  // proof.
  std::string Compress() const;
  static BacktrackingProof Decompress(const std::string &gz);

  void Append(uint8_t dfs_constraints_size, uint32_t mask, uint32_t query_elem,
              uint32_t store_elem) {
    dfs_constraints_size_array.push_back(dfs_constraints_size);
    mask_array.push_back(mask);
    query_elem_array.push_back(query_elem);
    store_elem_array.push_back(store_elem);
  }
  void Append(const BacktrackingProof &other);

  void Check() const {
    size_t n = dfs_constraints_size_array.size();
    CHECK_EQ(n, mask_array.size());
    CHECK_EQ(n, query_elem_array.size());
    CHECK_EQ(n, store_elem_array.size());
  }
  size_t Size() const { return dfs_constraints_size_array.size(); }
  bool Empty() const { return dfs_constraints_size_array.empty(); }
  void Reserve(size_t n) {
    dfs_constraints_size_array.reserve(n);
    mask_array.reserve(n);
    query_elem_array.reserve(n);
    store_elem_array.reserve(n);
  }

  std::vector<uint8_t> dfs_constraints_size_array;
  std::vector<uint32_t> mask_array;
  std::vector<uint32_t> query_elem_array;
  std::vector<uint32_t> store_elem_array;
};

// All per-orbit backtracking traces for one certificate, indexed by the dense
// orbit index. Mutated only serially (the prover writes it at each checkpoint);
// read concurrently during verification (const access is thread-safe).
//
// Each orbit's trace is stored gzip-compressed in `compressed`; presence is
// `!compressed[index].empty()`. A backtracking trace is never legitimately
// empty (the search only emits one when it improves the bound), so an empty
// blob is exactly an orbit with no backtracking proof.
struct BacktrackingProofArchive {
  // Number of orbit slots (== the certificate's constrained_tensors_size()).
  void Resize(size_t n) { compressed.resize(n); }
  size_t Size() const { return compressed.size(); }

  // Decompresses the orbit's trace on the fly (returns by value). An absent
  // slot
  // ("" blob) decompresses to a default/empty proof.
  BacktrackingProof Get(int index) const {
    CHECK_GE(index, 0);
    CHECK_LT(static_cast<size_t>(index), compressed.size());
    return BacktrackingProof::Decompress(compressed[index]);
  }
  // Compresses `proof` and stores it for `index`. An empty proof compresses to
  // "" and so clears the slot (the orbit has no backtracking proof), since
  // presence is `!compressed[index].empty()`.
  void Set(int index, const BacktrackingProof &proof) {
    CHECK_GE(index, 0);
    CHECK_LT(static_cast<size_t>(index), compressed.size());
    compressed[index] = proof.Compress();
  }
  void Clear(int index) { Set(index, BacktrackingProof{}); }

  // Deterministic, atomic (write-to-temp + rename) archive of the
  // length-prefixed compressed blobs (no whole-file gzip). No-op when `path` is
  // empty.
  void Save(const std::string &path) const;
  static BacktrackingProofArchive Load(const std::string &path);

  std::vector<std::string> compressed;
};
