#include "core/backtracking_proof.h"

#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <string>

#include "boost/iostreams/device/array.hpp"
#include "boost/iostreams/device/back_inserter.hpp"
#include "boost/iostreams/device/file.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "ng-log/logging.h"

namespace io = boost::iostreams;

namespace {
constexpr char kArchiveMagic[8] = {'B', 'T', 'P', 'A', 'R', 'C', 'H', '\0'};
constexpr uint32_t kArchiveVersion = 2;
} // namespace

std::string GetBacktrackingProofArchivePath(const std::string &proto_path) {
  std::string prefix;
  if (proto_path.ends_with(".pb.txt")) {
    prefix = proto_path.substr(0, proto_path.size() - 7);
  } else if (proto_path.ends_with(".pb")) {
    prefix = proto_path.substr(0, proto_path.size() - 3);
  } else {
    LOG(FATAL) << "Unsupported file extension: " << proto_path;
  }
  return prefix + ".btp";
}

void BacktrackingProof::WriteBody(std::ostream &out) const {
  Check();
  static_assert(std::endian::native == std::endian::little);
  const uint64_t n = static_cast<uint64_t>(Size());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  CHECK(out);
  if (n > 0) {
    out.write(reinterpret_cast<const char *>(dfs_constraints_size_array.data()),
              n * sizeof(uint8_t));
    out.write(reinterpret_cast<const char *>(mask_array.data()),
              n * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(query_elem_array.data()),
              n * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(store_elem_array.data()),
              n * sizeof(uint32_t));
    CHECK(out);
  }
}

BacktrackingProof BacktrackingProof::ReadBody(std::istream &in) {
  BacktrackingProof proof;
  static_assert(std::endian::native == std::endian::little);
  uint64_t n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  CHECK(in);
  proof.dfs_constraints_size_array.resize(static_cast<size_t>(n));
  proof.mask_array.resize(static_cast<size_t>(n));
  proof.query_elem_array.resize(static_cast<size_t>(n));
  proof.store_elem_array.resize(static_cast<size_t>(n));
  if (n > 0) {
    in.read(reinterpret_cast<char *>(proof.dfs_constraints_size_array.data()),
            n * sizeof(uint8_t));
    in.read(reinterpret_cast<char *>(proof.mask_array.data()),
            n * sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(proof.query_elem_array.data()),
            n * sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(proof.store_elem_array.data()),
            n * sizeof(uint32_t));
    CHECK(in);
  }
  proof.Check();
  return proof;
}

std::string BacktrackingProof::Compress() const {
  // An empty proof compresses to an empty string (an absent archive slot), so
  // Compress/Decompress round-trips it without a non-empty gzip header.
  if (Empty()) {
    return {};
  }
  std::string gz;
  {
    io::filtering_ostream out;
    out.push(io::gzip_compressor());
    out.push(io::back_inserter(gz));
    WriteBody(out);
    io::close(out);
  }
  return gz;
}

BacktrackingProof BacktrackingProof::Decompress(const std::string &gz) {
  if (gz.empty()) {
    return {};
  }
  io::filtering_istream in;
  in.push(io::gzip_decompressor());
  in.push(io::array_source(gz.data(), gz.size()));
  return ReadBody(in);
}

void BacktrackingProof::Save(const std::string &path) const {
  if (path.empty()) {
    return;
  }
  io::filtering_ostream out;
  out.push(io::gzip_compressor());
  out.push(io::file_sink(path));
  CHECK(out) << "Failed to open file for writing: " << path;
  WriteBody(out);
  io::close(out);
}

BacktrackingProof BacktrackingProof::Load(const std::string &path) {
  io::filtering_istream in;
  in.push(io::gzip_decompressor());
  in.push(io::file_source(path));
  CHECK(in) << "Failed to open file for reading: " << path;
  return ReadBody(in);
}

void BacktrackingProof::Append(const BacktrackingProof &other) {
  dfs_constraints_size_array.insert(dfs_constraints_size_array.end(),
                                    other.dfs_constraints_size_array.begin(),
                                    other.dfs_constraints_size_array.end());
  mask_array.insert(mask_array.end(), other.mask_array.begin(),
                    other.mask_array.end());
  query_elem_array.insert(query_elem_array.end(),
                          other.query_elem_array.begin(),
                          other.query_elem_array.end());
  store_elem_array.insert(store_elem_array.end(),
                          other.store_elem_array.begin(),
                          other.store_elem_array.end());
}

void BacktrackingProofArchive::Save(const std::string &path) const {
  if (path.empty()) {
    return;
  }
  static_assert(std::endian::native == std::endian::little);
  const uint64_t count = static_cast<uint64_t>(compressed.size());

  // Plain file: header, then `count` length-prefixed compressed blobs in
  // dense-index order. The blobs are already gzip-compressed (per proof), so
  // the file is never compressed as a whole and is written by moving bytes
  // only.
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary);
    CHECK(out) << "Failed to open file for writing: " << tmp;
    out.write(kArchiveMagic, sizeof(kArchiveMagic));
    out.write(reinterpret_cast<const char *>(&kArchiveVersion),
              sizeof(kArchiveVersion));
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const std::string &blob : compressed) {
      const uint64_t len = static_cast<uint64_t>(blob.size());
      out.write(reinterpret_cast<const char *>(&len), sizeof(len));
      if (len > 0) {
        out.write(blob.data(), static_cast<std::streamsize>(len));
      }
    }
    CHECK(out);
  }
  std::filesystem::rename(tmp, path);
}

BacktrackingProofArchive
BacktrackingProofArchive::Load(const std::string &path) {
  BacktrackingProofArchive archive;
  std::ifstream in(path, std::ios::binary);
  CHECK(in) << "Failed to open file for reading: " << path;
  static_assert(std::endian::native == std::endian::little);

  char magic[sizeof(kArchiveMagic)];
  in.read(magic, sizeof(magic));
  CHECK(in);
  CHECK_EQ(std::memcmp(magic, kArchiveMagic, sizeof(magic)), 0)
      << "bad backtracking archive magic: " << path;
  uint32_t version = 0;
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  CHECK_EQ(version, kArchiveVersion) << "unsupported archive version: " << path;
  uint64_t count = 0;
  in.read(reinterpret_cast<char *>(&count), sizeof(count));
  CHECK(in);

  // Read the compressed blobs verbatim (no decompression here — Get
  // decompresses a single proof on demand), so the archive is held in memory
  // compressed.
  archive.Resize(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t len = 0;
    in.read(reinterpret_cast<char *>(&len), sizeof(len));
    CHECK(in);
    if (len > 0) {
      archive.compressed[i].resize(static_cast<size_t>(len));
      in.read(archive.compressed[i].data(), static_cast<std::streamsize>(len));
      CHECK(in);
    }
  }
  return archive;
}
