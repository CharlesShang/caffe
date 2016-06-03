#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/multiple_inner_product_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"


#define WIDTH  2
#define HEIGHT 2
#define CHENNAL 2
#define NUM 5

#define NUM_LAYER 3

#define NUM_OUT1 3
#define NUM_OUT2 2
#define NUM_OUT3 4

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class MultipleInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultipleInnerProductLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),       
        blob_top_(new Blob<Dtype>()){}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    vector<int> shape1;
    shape1.push_back(NUM);
    shape1.push_back(CHENNAL);
    shape1.push_back(HEIGHT);
    shape1.push_back(WIDTH);
    blob_bottom_->Reshape(shape1);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler1(filler_param);
    GaussianFiller<Dtype> filler2(filler_param);    
    filler1.Fill(this->blob_bottom_);
    // for (int i = 0; i < NUM; ++i){
    //   for (int j = 0; j < CHENNAL * HEIGHT * WIDTH; ++j){
    //     int idx = i * CHENNAL * HEIGHT * WIDTH + j;
    //     blob_bottom_->mutable_cpu_data()[idx] = i * 10;
    //   } 
    // }

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

  }
  inline string blob_to_string(Blob<Dtype> *data) const {
    ostringstream stream;
    for (int i = 0; i < data->count(); ++i) {
      stream << data->mutable_cpu_data()[i] << " ";
    }
    return stream.str();
  }

  virtual ~MultipleInnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(MultipleInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultipleInnerProductLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(3);

  mip_param->add_num_outputs(NUM_OUT1);
  mip_param->add_num_outputs(NUM_OUT2);
  mip_param->add_num_outputs(NUM_OUT3);

  MultipleInnerProductLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for(int i = 0; i < this->blob_top_vec_.size(); i++)
  {
    Blob<Dtype>* blob = this->blob_top_vec_[i];
    EXPECT_EQ(blob->num(), NUM);
    EXPECT_EQ(blob->channels(), NUM_OUT3);
    EXPECT_EQ(blob->height(), 1);
    EXPECT_EQ(blob->width(), 1);
  }
  EXPECT_EQ(layer.blobs().size(), 3 * 2);
  EXPECT_EQ(layer.blobs()[0]->shape()[0], NUM_OUT1);
  EXPECT_EQ(layer.blobs()[2]->shape()[0], NUM_OUT2);
  EXPECT_EQ(layer.blobs()[4]->shape()[0], NUM_OUT3);

}

TYPED_TEST(MultipleInnerProductLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(NUM_LAYER);

  mip_param->add_num_outputs(NUM_OUT1);
  mip_param->add_num_outputs(NUM_OUT2);
  mip_param->add_num_outputs(NUM_OUT3);

  MultipleInnerProductLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
}

TYPED_TEST(MultipleInnerProductLayerTest, MultipleInnerProductLayerTestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(3);
  mip_param->set_activation("None");  // 

  mip_param->add_num_outputs(NUM_OUT1);
  mip_param->add_num_outputs(NUM_OUT2);
  mip_param->add_num_outputs(NUM_OUT3);

  MultipleInnerProductLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2);

  // coz MIP has ReLU layer, gradient checker wont pass :-)
  // comment the relu layer in multiple_inner_product_layer, this checker will pass
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(MultipleInnerProductLayerTest, TestGradientSimple) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(1);

  mip_param->add_num_outputs(NUM_OUT1);
  // mip_param->add_num_outputs(NUM_OUT1);

  MultipleInnerProductLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2); // step_size, threshold

  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(MultipleInnerProductLayerTest, MultipleInnerProductLayerTestLoadWieghtsFromFile) {
  typedef typename TypeParam::Dtype Dtype;
  
  const int num_layer = 5;
  const int num_outputs[] = {3,4,5,6,7};

  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(num_layer);
  for (int i = 0; i < num_layer; ++i){
    mip_param->add_num_outputs(num_outputs[i]);
  }

  MultipleInnerProductLayer<Dtype> layer(layer_param);

  
  layer.blobs().resize(num_layer * 2);
  
  vector<int> weight_shape(2, 0);
  weight_shape[0] = CHENNAL*HEIGHT*WIDTH; // in
  vector<int> bias_shape(1, 0);
  layer.blobs().resize(num_layer * 2);
  for (int i = 0; i < num_layer; i ++){
    weight_shape[1] = weight_shape[0];
    weight_shape[0] = num_outputs[i];
    bias_shape[0] = num_outputs[i];
    layer.blobs()[i * 2].reset(new Blob<Dtype>(weight_shape));
    layer.blobs()[i * 2 + 1].reset(new Blob<Dtype>(bias_shape));
  }
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<Dtype>* blob = this->blob_top_vec_[0];
  EXPECT_EQ(blob->num(), NUM);
  EXPECT_EQ(blob->channels(), num_outputs[num_layer - 1]);
  EXPECT_EQ(blob->height(), 1);
  EXPECT_EQ(blob->width(), 1);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(MultipleInnerProductLayerTest, MultipleInnerProductLayerTestBenchmarking) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(3);
  mip_param->set_activation("None");  // 

  mip_param->add_num_outputs(NUM_OUT1);
  mip_param->add_num_outputs(NUM_OUT2);
  mip_param->add_num_outputs(NUM_OUT3);

  MultipleInnerProductLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2);

  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);

  using caffe::Timer;
  Timer total_timer;
  Timer forward_timer;
  Timer backward_timer;
  double forward_time = 0.0;
  double backward_time = 0.0;
  total_timer.Start();
  
  
  for (int i = 0; i < 10; ++i){
    forward_timer.Start();
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    forward_time += forward_timer.MilliSeconds();
    backward_timer.Start();
    layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
    backward_time += backward_timer.MilliSeconds();
  }
  total_timer.Stop();
  LOG(ERROR) << "Total    Time: " << total_timer.MilliSeconds() / 10.0 << " ms.";
  LOG(ERROR) << "Forward  Time: " << forward_time / 10.0  << " ms.";
  LOG(ERROR) << "Backward Time: " << backward_time / 10.0 << " ms.";

}

TYPED_TEST(MultipleInnerProductLayerTest, MultipleInnerProductLayerTestCompute) {
  typedef typename TypeParam::Dtype Dtype;
  
  const int num_layer = 2;
  const int num_output = 5;
  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(num_layer);
  mip_param->add_num_outputs(num_output);
  mip_param->add_num_outputs(num_output);
  mip_param->set_activation("None");  
  

  MultipleInnerProductLayer<Dtype> layer(layer_param);

  vector<int> weight_shape(2, num_output);
  vector<int> bias_shape(1, num_output);
  layer.blobs().resize(num_layer * 2);
  layer.blobs()[0].reset(new Blob<Dtype>(weight_shape));
  layer.blobs()[1].reset(new Blob<Dtype>(bias_shape));
  layer.blobs()[2].reset(new Blob<Dtype>(weight_shape));
  layer.blobs()[3].reset(new Blob<Dtype>(bias_shape));
  // eye matrix
  for (int i = 0; i < layer.blobs()[0]->shape()[0]; ++i){
    int spatial_dim = layer.blobs()[0]->shape()[1];
    for (int j = 0; j < layer.blobs()[0]->shape()[1]; ++j){
      int idx = i * spatial_dim + j;
      // float v = (i == j ? 1 : 0);      // eye
      float v = float(caffe_rng_rand()) / UINT_MAX; // random matrix
      layer.blobs()[0]->mutable_cpu_data()[idx] = v;
      layer.blobs()[2]->mutable_cpu_data()[idx] = v;
    }
  }
  for (int i = 0; i < num_output; ++i){
    layer.blobs()[1]->mutable_cpu_data()[i] = 0;
    layer.blobs()[3]->mutable_cpu_data()[i] = 0;
  }

  // LOG(ERROR) <<"Weight: " << this->blob_to_string(layer.blobs()[0].get());
  // LOG(ERROR) <<"Bias:   " << this->blob_to_string(layer.blobs()[1].get());

  vector<Blob<Dtype>*> blob_bottom_vec(1, NULL), blob_top_vec(1, NULL);
  shared_ptr<Blob<Dtype> > blob_bottom, blob_top;
  blob_bottom.reset(new Blob<Dtype>(1, num_output, 1, 1));
  blob_top.reset(new Blob<Dtype>());
  blob_bottom_vec[0] = blob_bottom.get();
  blob_top_vec[0]    = blob_top.get();

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler1(filler_param);
  filler1.Fill(blob_bottom.get());

  LOG(ERROR) <<"Bottom: " << this->blob_to_string(blob_bottom.get());

  layer.Reshape(blob_bottom_vec, blob_top_vec);
  
  layer.Forward(blob_bottom_vec, blob_top_vec);

  LOG(ERROR) <<"Top:    " << this->blob_to_string(blob_top.get());
  // Top = Bottom * W' * W'

}

}  // namespace caffe
