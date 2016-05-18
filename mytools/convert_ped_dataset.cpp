// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] IMAGEFOLD DMAPFOLD WINDOWS_FILE IMG_DB_NAME DMAP_DB_NAME

#include "common_tools.h"

using namespace std;
using namespace cv;
//using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using boost::filesystem::path;
using caffe::Datum;
namespace db = caffe::db;

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(prefix, "",
        "image dir");
DEFINE_int32(resize_width, 80, "Width images are resized to");
DEFINE_int32(resize_height, 160, "Height images are resized to");
// DEFINE_string(img_db, "img_db",
//         "The backend {lmdb, leveldb} for storing the result");
// DEFINE_string(dmap_db, "dmap_db",
//         "The backend {lmdb, leveldb} for storing the result");
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_dataset ANNOTATINFILE DBPATH\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    cout << argc <<endl;
    gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_dataset");
    return 1;
  }

  const int resize_width = FLAGS_resize_width;
  const int resize_height = FLAGS_resize_height;
  const string backend = FLAGS_backend;
  const string prefix = FLAGS_prefix;

  string annotation = argv[1];
  string database = argv[2];

  LOG(INFO) << "annotation " << annotation;
  LOG(INFO) << "database " << database;

 // train
  convert_img_data_ped(prefix, annotation, 3, resize_width, resize_height, database + "/train", backend, 0, 0.8);

  // val
  convert_img_data_ped(prefix, annotation, 3, resize_width, resize_height, database + "/val", backend, 0.8, 1.0);
}