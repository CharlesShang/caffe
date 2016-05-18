#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/clustering_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define K 3

#define WIDTH  2
#define HEIGHT 2
#define CHENNAL 1
#define NUM 300
#define NUM_OUT 3

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class ClusteringLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ClusteringLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom1_(new Blob<Dtype>()),
        blob_bottom2_(new Blob<Dtype>()),
        
        blob_top_(new Blob<Dtype>()){}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    vector<int> shape1, shape2;
    shape1.push_back(NUM);
    shape1.push_back(CHENNAL);
    shape1.push_back(HEIGHT);
    shape1.push_back(WIDTH);
    shape2.push_back(NUM);
    shape2.push_back(1);
    shape2.push_back(1);
    shape2.push_back(1);
    blob_bottom1_->Reshape(shape1);
    blob_bottom2_->Reshape(shape2);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler1(filler_param);
    GaussianFiller<Dtype> filler2(filler_param);    
    filler1.Fill(this->blob_bottom1_);
    for (int i = 0; i < NUM; ++i){
      blob_bottom2_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }
    for (int i = 0; i < NUM; ++i){
      for (int j = 0; j < CHENNAL * HEIGHT * WIDTH; ++j){
        int idx = i * CHENNAL * HEIGHT * WIDTH + j;
        blob_bottom1_->mutable_cpu_data()[idx] = i * 10;
      } 
    }

    blob_bottom_vec_.push_back(blob_bottom1_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);

  }
  inline string blob_to_string(Blob<Dtype> *data) const {
    ostringstream stream;
    for (int i = 0; i < data->count(); ++i) {
      stream << data->mutable_cpu_data()[i] << " ";
    }
    return stream.str();
  }

  virtual ~ClusteringLayerTest() {
    delete blob_bottom1_;
    delete blob_bottom2_;
    delete blob_top_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(ClusteringLayerTest, TestDtypesAndDevices);

TYPED_TEST(ClusteringLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusteringParameter* clustering_layer_param = layer_param.mutable_clustering_param();
  clustering_layer_param->set_num_output(NUM_OUT);
  clustering_layer_param->set_total_class(2);

  ClusteringLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for(int i = 0; i < this->blob_top_vec_.size(); i++)
  {
    Blob<Dtype>* blob_ = this->blob_top_vec_[i];
    EXPECT_EQ(blob_->num(), NUM);
    EXPECT_EQ(blob_->channels(), NUM_OUT);
    EXPECT_EQ(blob_->height(), 1);
    EXPECT_EQ(blob_->width(), 1);
  }
}

TYPED_TEST(ClusteringLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusteringParameter* clustering_layer_param = layer_param.mutable_clustering_param();
  clustering_layer_param->set_num_output(NUM_OUT);
  clustering_layer_param->set_total_class(2);

  ClusteringLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
}

TYPED_TEST(ClusteringLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusteringParameter* clustering_layer_param = layer_param.mutable_clustering_param();
  clustering_layer_param->set_num_output(NUM_OUT);
  clustering_layer_param->set_total_class(2);
  clustering_layer_param->set_branch(false); 
  clustering_layer_param->set_across_class(false); 
  ClusteringLayer<Dtype> layer(layer_param);
  // GradientChecker<Dtype> checker(1e-2, 1e-2);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // let the layer caching enough data
  // do kmeans 
  for (int i = 0; i < 10; ++i){
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  // LOG(ERROR) << " check point " ;
  // getchar();
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ClusteringLayerTest, TestGradientACROSSLABEL) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusteringParameter* clustering_layer_param = layer_param.mutable_clustering_param();
  clustering_layer_param->set_num_output(NUM_OUT);
  clustering_layer_param->set_total_class(3);
  clustering_layer_param->set_branch(false); 
  clustering_layer_param->set_across_class(false); 

  ClusteringLayer<Dtype> layer(layer_param);
  // GradientChecker<Dtype> checker(1e-2, 1e-2);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // let the layer caching enough data
  // do kmeans 
  for (int i = 0; i < 10; ++i){
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  // LOG(ERROR) << " check point " ;
  // getchar();
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ClusteringLayerTest, TestReshapeForwardForTest) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusteringParameter* clustering_layer_param = layer_param.mutable_clustering_param();
  clustering_layer_param->set_num_output(NUM_OUT);
  clustering_layer_param->set_total_class(1);
  clustering_layer_param->set_branch(false); 
  clustering_layer_param->set_across_class(true); 
  clustering_layer_param->set_k(K); 

  ClusteringLayer<Dtype> layer(layer_param);
  layer.blobs().resize(K * 3);
  vector<int> weight_shape(2);
  weight_shape[0] = NUM_OUT;
  weight_shape[1] = CHENNAL*HEIGHT*WIDTH;
  vector<int> bias_shape(1, NUM_OUT);
  for (int i = 0; i < K * 2; ){
    layer.blobs()[i++].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Blob<Dtype> > & blob = layer.blobs()[i-1];
    for (int c = 0; c < blob->count(); ++c){
      blob->mutable_cpu_data()[c] = 1;
    }
    layer.blobs()[i++].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Blob<Dtype> > & blob1 = layer.blobs()[i-1];
    for (int c = 0; c < blob1->count(); ++c){
      blob1->mutable_cpu_data()[c] = 0;
    }
  }
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = CHENNAL;
  shape[2] = HEIGHT;
  shape[3] = WIDTH;
  for (int i = K * 2, v = 1; i < K * 3; v ++, i ++){
    layer.blobs()[i].reset(new Blob<Dtype>(shape));
    shared_ptr<Blob<Dtype> > & blob = layer.blobs()[i];
    for (int c = 0; c < blob->count(); ++c){
      blob->mutable_cpu_data()[c] = v;
    }
  }
  
  for (int i = 0; i < NUM; ++i){
    for (int j = 0; j < CHENNAL * HEIGHT * WIDTH; ++j){
      int idx = i * CHENNAL * HEIGHT * WIDTH + j;
      this->blob_bottom1_->mutable_cpu_data()[idx] = caffe_rng_rand() % 10;
    } 
  }

  // let the layer caching enough data
  // do kmeans 
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);
  for (int i = 0; i < 10; ++i){
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }

  LOG(ERROR) << this->blob_to_string(this->blob_bottom_vec_[0]);
  LOG(ERROR) << this->blob_to_string(this->blob_top_vec_[0]);
  // LOG(ERROR) << " check point " ;
  // getchar();
}

TYPED_TEST(ClusteringLayerTest, ClusteringLayerTestKmeans) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClusteringParameter* clustering_layer_param = layer_param.mutable_clustering_param();
  int total_class = 2;
  int k = 3;
  clustering_layer_param->set_num_output(NUM_OUT);
  clustering_layer_param->set_total_class(total_class);
  clustering_layer_param->set_k(k);
  clustering_layer_param->set_branch(true); 
  clustering_layer_param->set_across_class(false); 

  ClusteringLayer<Dtype> layer(layer_param);
 
  // set data
  // this->blob_bottom1_ = new Blob<Dtype>(50, CHENNAL, HEIGHT, WIDTH);
  // this->blob_bottom2_ = new Blob<Dtype>(50, 1, 1, 1);
  for (int n = 0; n < NUM; ++n)
  {
    float r = caffe_rng_rand() % 3 * 10;
    for (int i = 0; i < CHENNAL * WIDTH * HEIGHT; ++i)
    {
      const int idx = n * CHENNAL * WIDTH * HEIGHT + i;
      this->blob_bottom1_->mutable_cpu_data()[idx] = r + 3.0 * caffe_rng_rand() / UINT_MAX;
    }
    this->blob_bottom2_->mutable_cpu_data()[n] = caffe_rng_rand() % 2;  // label
  }
  this->blob_bottom_vec_.resize(2);
  this->blob_bottom_vec_[0] = this->blob_bottom1_;
  this->blob_bottom_vec_[1] = this->blob_bottom2_;

  // do kmeans
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // checking
  vector<shared_ptr<Blob<Dtype> > > & blobs = layer.blobs();
  EXPECT_EQ(blobs.size(), total_class * k * 3);

  // LOG(ERROR) << "Blob size: " << UINT_MAX;

  // LOG(ERROR) << "Blob size: " << blobs.size();
  for (int i = total_class * k * 2; i < blobs.size(); ++i)
  {
    LOG(ERROR) << this->blob_to_string(blobs[i].get());
  }
  
}

}  // namespace caffe
