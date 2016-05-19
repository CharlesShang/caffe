# ifndef COMMON_TOOLS
# define COMMON_TOOLS
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <string.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include "google/protobuf/text_format.h"
#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
#include "leveldb/db.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;
using namespace cv;
using caffe::Datum;
using boost::scoped_ptr;
using std::pair;
namespace db = caffe::db;

const int CHANNELS_DIM_IMG = 3;
const int INPUT_DIM_IMG = 100;
const int BUFFER_SIZE_IMG = INPUT_DIM_IMG * INPUT_DIM_IMG * CHANNELS_DIM_IMG;

const int CHANNELS_DIM_HEATMAP = 1;
const int INPUT_DIM_HEATMAP = 32;
const int BUFFER_SIZE_HEATMAP = INPUT_DIM_HEATMAP * INPUT_DIM_HEATMAP * CHANNELS_DIM_HEATMAP;

const float train_ratio = 0.5;
const float val_ratio   = 0.1;
const float test_ratio  = 0.4;

struct box{
  int x1, y1, x2, y2;
};
typedef struct box box;
struct box_num{
  float num;
  box b;
  string str(){
  	stringstream ss;
  	ss << num << ":" << b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2;
  	return ss.str();
  };
};
typedef struct box_num box_num;

// read text format data into boxesnums and img_files
int read_in_boxesnums(string windows_path,
  vector < vector<box_num> > &boxesnums, vector<string> &img_files);

// crop img --> c*h*w str_buffer_img
void crop_sample_in_image(IplImage *img, box b, int normed_width, int normed_height, unsigned char *str_buffer_img);

// crop m --> h*w data  (m.size==raws, m[0].size==cols)
void crop_in_densitymap(vector< vector<float> > &m, box b, int normed_width, int normed_height, float *data);

void read_in_densitymap(string windows_file, vector< vector<float> > &m);

string get_file_stem(string &file_path);
void split_path(string &file_path, string &path, string &name, string &ext);

void save_dmap_to_datum(float * dmap, int w, int h, Datum &datum);

void convert_img_data(string windows_file_path, string image_fold_path, int c, int resize_width, int resize_height, string db_path, string backend = "lmdb", float begin_ratio = 0, float end_ratio = 0.01, bool check_size = false);

void convert_float_label_data(string windows_file_path, string image_fold_path, string db_path, string backend = "lmdb", float begin_ratio = 0, float end_ratio = 0.01, bool check_size = false);

void convert_dmap_data(string windows_file_path, string dmap_fold_path, int c, int dmap_width, int dmap_height, string db_path, string backend = "lmdb", float begin_ratio = 0, float end_ratio = 0.01, bool check_size = false);

int read_in_boxage_ifa(string annotation,
  vector<box_num> &boxage, vector<string> &img_files);
int read_in_boxage_ifa_morph(string annotation,
  vector<box_num> &boxage, vector<string> &img_files);
void crop_sample_in_image_ifa(IplImage *img, box b, int normed_width, int normed_height, unsigned char *str_buffer_img);
void convert_img_data_ifa(string annotation, int c, int resize_width, int resize_height, string db_path, string backend = "lmdb", float begin_ratio = 0, float end_ratio = 0.01, bool check_size = false);
void data_augment_ifa(IplImage *img, box b, int normed_width, int normed_height, vector<unsigned char*> &str_buffers);
vector<string> SplitCSVIntoTokens(string line);

// pedestrian
void convert_img_data_ped(string prefix, 
	string annotation, 
	int c, int resize_width, int resize_height, 
	string db_path, string backend = "lmdb", 
	float begin_ratio = 0, float end_ratio = 0.01);
int read_in_annotations_ped(string annotation,
  vector<vector<box_num> > &box_ids, vector<vector<box_num> > &box_gts, 
  vector<string> &img_files);
void crop_sample_in_image(IplImage *img, box b, 
	int normed_width, int normed_height, 
	unsigned char *str_buffer_img);
#endif
