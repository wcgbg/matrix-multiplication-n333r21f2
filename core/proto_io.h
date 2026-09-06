#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "boost/iostreams/device/file.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "ng-log/logging.h"

// Read a proto message from a file. Supported extensions:
//   .pb.txt       text format
//   .pb           binary
//   .pb.txt.gz    gzip-compressed text format
//   .pb.gz        gzip-compressed binary
template <typename T> T ReadProtoFromFile(const std::string &path) {
  LOG(INFO) << "Reading from " << path << " ...";
  T message;
  if (path.ends_with(".pb.txt.gz")) {
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(boost::iostreams::file_source(path, std::ios::binary));
    CHECK(in);
    google::protobuf::io::IstreamInputStream is(&in);
    CHECK(google::protobuf::TextFormat::Parse(&is, &message));
  } else if (path.ends_with(".pb.gz")) {
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(boost::iostreams::file_source(path, std::ios::binary));
    CHECK(in);
    CHECK(message.ParseFromIstream(&in));
  } else if (path.ends_with(".pb.txt")) {
    std::ifstream in(path);
    CHECK(in);
    google::protobuf::io::IstreamInputStream is(&in);
    CHECK(google::protobuf::TextFormat::Parse(&is, &message));
  } else if (path.ends_with(".pb")) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in);
    CHECK(message.ParseFromIstream(&in));
  } else {
    LOG(FATAL) << "Unsupported file extension: " << path;
  }
  LOG(INFO) << "Read " << message.ByteSizeLong() << " bytes";
  return message;
}

// Write a proto message to a file (.pb.txt = text format, .pb = binary).
template <typename T>
void WriteProtoToFile(const T &message, const std::string &path,
                      bool also_write_to_txt = false, bool quiet = false) {
  if (!quiet) {
    LOG(INFO) << "Writing to " << path << " ...";
  }
  const std::string tmp_path = path + ".tmp";
  if (path.ends_with(".pb.txt")) {
    std::ofstream out(tmp_path);
    CHECK(out) << "Failed to open file: " << tmp_path;
    google::protobuf::io::OstreamOutputStream os(&out);
    CHECK(google::protobuf::TextFormat::Print(message, &os));
  } else if (path.ends_with(".pb")) {
    std::ofstream out(tmp_path, std::ios::binary);
    CHECK(out) << "Failed to open file: " << tmp_path;
    CHECK(message.SerializeToOstream(&out));
  } else {
    LOG(FATAL) << "Unsupported file extension: " << path;
  }
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
  std::filesystem::rename(tmp_path, path);
  if (path.ends_with(".pb") && also_write_to_txt) {
    WriteProtoToFile(message, path + ".txt");
  }
  if (!quiet) {
    LOG(INFO) << "Written " << message.ByteSizeLong() << " bytes";
  }
}
