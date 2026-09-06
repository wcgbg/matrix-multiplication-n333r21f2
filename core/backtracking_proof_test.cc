#include "core/backtracking_proof.h"

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

namespace {

TEST(BacktrackingProofTest, SaveLoadRoundTripEmpty) {
  const std::string temp_dir = testing::TempDir();
  const std::string path = temp_dir + "/backtracking_proof_empty.gz";

  BacktrackingProof proof;
  proof.Check();
  EXPECT_EQ(proof.Size(), 0u);
  proof.Save(path);
  BacktrackingProof loaded = BacktrackingProof::Load(path);
  EXPECT_EQ(loaded.Size(), 0u);
  EXPECT_TRUE(loaded.dfs_constraints_size_array.empty());
  EXPECT_TRUE(loaded.mask_array.empty());
  EXPECT_TRUE(loaded.query_elem_array.empty());
  EXPECT_TRUE(loaded.store_elem_array.empty());
}

TEST(BacktrackingProofTest, SaveLoadRoundTripNonEmpty) {
  const std::string temp_dir = testing::TempDir();
  const std::string path = temp_dir + "/backtracking_proof_nonempty.gz";

  BacktrackingProof proof;
  for (int i = 0; i < 100; i++) {
    proof.Append(static_cast<uint8_t>(i), i * 2u, i * 4u, i * 5u);
  }
  proof.Check();
  EXPECT_EQ(proof.Size(), 100u);
  proof.Save(path);

  BacktrackingProof loaded = BacktrackingProof::Load(path);
  ASSERT_EQ(loaded.Size(), proof.Size());
  for (size_t i = 0; i < proof.Size(); i++) {
    EXPECT_EQ(loaded.dfs_constraints_size_array[i],
              proof.dfs_constraints_size_array[i]);
    EXPECT_EQ(loaded.mask_array[i], proof.mask_array[i]);
    EXPECT_EQ(loaded.query_elem_array[i], proof.query_elem_array[i]);
    EXPECT_EQ(loaded.store_elem_array[i], proof.store_elem_array[i]);
  }
}

TEST(BacktrackingProofTest, AppendOther) {
  BacktrackingProof a;
  a.Append(1, 100u, 1u, 2u);
  BacktrackingProof b;
  b.Append(2, 200u, 3u, 4u);
  b.Append(3, 300u, 5u, 6u);
  a.Append(b);
  EXPECT_EQ(a.Size(), 3u);
  EXPECT_EQ(a.dfs_constraints_size_array[0], 1);
  EXPECT_EQ(a.dfs_constraints_size_array[1], 2);
  EXPECT_EQ(a.dfs_constraints_size_array[2], 3);
  EXPECT_EQ(a.mask_array[0], 100u);
  EXPECT_EQ(a.mask_array[1], 200u);
  EXPECT_EQ(a.mask_array[2], 300u);
  EXPECT_EQ(a.query_elem_array[0], 1u);
  EXPECT_EQ(a.query_elem_array[1], 3u);
  EXPECT_EQ(a.query_elem_array[2], 5u);
  EXPECT_EQ(a.store_elem_array[0], 2u);
  EXPECT_EQ(a.store_elem_array[1], 4u);
  EXPECT_EQ(a.store_elem_array[2], 6u);
}

TEST(BacktrackingProofTest, GetBacktrackingProofArchivePath) {
  EXPECT_EQ(GetBacktrackingProofArchivePath("cert_n3.pb.txt"), "cert_n3.btp");
  EXPECT_EQ(GetBacktrackingProofArchivePath("proof/cert_n4.pb"),
            "proof/cert_n4.btp");
}

namespace {
BacktrackingProof MakeProof(int seed, size_t entries) {
  BacktrackingProof proof;
  for (size_t i = 0; i < entries; ++i) {
    proof.Append(static_cast<uint8_t>(seed + i), (seed + i) * 7u,
                 (seed + i) * 11u, (seed + i) * 13u);
  }
  return proof;
}

void ExpectProofEq(const BacktrackingProof &a, const BacktrackingProof &b) {
  ASSERT_EQ(a.Size(), b.Size());
  for (size_t i = 0; i < a.Size(); ++i) {
    EXPECT_EQ(a.dfs_constraints_size_array[i], b.dfs_constraints_size_array[i]);
    EXPECT_EQ(a.mask_array[i], b.mask_array[i]);
    EXPECT_EQ(a.query_elem_array[i], b.query_elem_array[i]);
    EXPECT_EQ(a.store_elem_array[i], b.store_elem_array[i]);
  }
}
} // namespace

TEST(BacktrackingProofArchiveTest, RoundTripMixedPresence) {
  const std::string path = testing::TempDir() + "/archive_mixed.btp";

  BacktrackingProofArchive archive;
  archive.Resize(5);
  // Leave slots 1 and 3 absent; populate 0, 2, 4.
  archive.Set(0, MakeProof(1, 3));
  archive.Set(2, MakeProof(20, 100));
  archive.Set(4, MakeProof(7, 1));
  archive.Save(path);

  BacktrackingProofArchive loaded = BacktrackingProofArchive::Load(path);
  EXPECT_EQ(loaded.Size(), 5u);
  // Presence is `!compressed[i].empty()`.
  EXPECT_FALSE(loaded.compressed[0].empty());
  EXPECT_TRUE(loaded.compressed[1].empty());
  EXPECT_FALSE(loaded.compressed[2].empty());
  EXPECT_TRUE(loaded.compressed[3].empty());
  EXPECT_FALSE(loaded.compressed[4].empty());
  ExpectProofEq(loaded.Get(0), MakeProof(1, 3));
  ExpectProofEq(loaded.Get(2), MakeProof(20, 100));
  ExpectProofEq(loaded.Get(4), MakeProof(7, 1));
}

TEST(BacktrackingProofArchiveTest, SetClear) {
  BacktrackingProofArchive archive;
  archive.Resize(3);
  EXPECT_TRUE(archive.compressed[0].empty());
  archive.Set(0, MakeProof(2, 4));
  EXPECT_FALSE(archive.compressed[0].empty());
  archive.Clear(0);
  EXPECT_TRUE(archive.compressed[0].empty());
  // Setting an empty proof also clears the slot.
  archive.Set(0, MakeProof(2, 4));
  archive.Set(0, BacktrackingProof{});
  EXPECT_TRUE(archive.compressed[0].empty());
}

TEST(BacktrackingProofArchiveTest, EmptyArchiveRoundTrip) {
  const std::string path = testing::TempDir() + "/archive_empty.btp";
  BacktrackingProofArchive archive;
  archive.Resize(4); // all absent
  archive.Save(path);
  BacktrackingProofArchive loaded = BacktrackingProofArchive::Load(path);
  EXPECT_EQ(loaded.Size(), 4u);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(loaded.compressed[i].empty());
  }
}

TEST(BacktrackingProofTest, CompressDecompressRoundTrip) {
  const BacktrackingProof proof = MakeProof(5, 200);
  const std::string gz = proof.Compress();
  EXPECT_FALSE(gz.empty());
  ExpectProofEq(BacktrackingProof::Decompress(gz), proof);

  // An empty proof compresses to "", and "" decompresses to a default proof.
  const BacktrackingProof empty;
  EXPECT_TRUE(empty.Compress().empty());
  ExpectProofEq(BacktrackingProof::Decompress(std::string{}), empty);
}

} // namespace
