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
DEFINE_int32(resize_width, 64, "Width images are resized to");
DEFINE_int32(resize_height, 64, "Height images are resized to");
DEFINE_int32(dmap_width, 18, "Width images are resized to");
DEFINE_int32(dmap_height, 18, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
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
        "    convert_dataset [FLAGS] IMAGEFOLD DMAPFOLD WINDOWSFILE IMG_DB_NAME DMAP_DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 6) {
    cout << argc <<endl;
    gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_dataset");
    return 1;
  }

  const bool check_size = FLAGS_check_size;
  const int resize_width = FLAGS_resize_width;
  const int resize_height = FLAGS_resize_height;
  const int dmap_width = FLAGS_dmap_width;
  const int dmap_height = FLAGS_dmap_height;
  const string backend = FLAGS_backend;

  string image_fold_path = argv[1];
  string dmap_fold_path = argv[2];
  string windows_file_path = argv[3];
  string db_path = argv[4];
  string db_path2 = argv[5];

  LOG(INFO) << "image_fold_path " << image_fold_path;
  LOG(INFO) << "dmap_fold_path " << dmap_fold_path;
  LOG(INFO) << "windows_file_path " << windows_file_path;
  LOG(INFO) << "db_path " << db_path;
  LOG(INFO) << "db_path2 " << db_path2;

 // train
  convert_dmap_data(windows_file_path, dmap_fold_path, 1, dmap_width, dmap_height, db_path2 + "_train", backend, 0, 0.5,  check_size);
  convert_img_data(windows_file_path, image_fold_path, 3, resize_width, resize_height, db_path + "_train", backend, 0, 0.5, check_size);
  convert_float_label_data(windows_file_path, image_fold_path, db_path + "_float_label_train", backend, 0, 0.5, check_size);

  // val
  convert_img_data(windows_file_path, image_fold_path, 3, resize_width, resize_height, db_path + "_val", backend, 0.5, 0.6, check_size);
  convert_dmap_data(windows_file_path, dmap_fold_path, 1, dmap_width, dmap_height, db_path2 + "_val", backend, 0.5, 0.6,  check_size);
  convert_float_label_data(windows_file_path, image_fold_path, db_path + "_float_label_val", backend, 0.5, 0.6, check_size);
}