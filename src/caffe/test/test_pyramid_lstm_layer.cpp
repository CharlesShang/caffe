#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/transpose_layer.hpp"
#include "caffe/layers/pyramid_lstm_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define NUM_CELLS 3
#define BATCH_SIZE 1
#define WIDTH  2
#define HEIGHT 2
#define INPUT_DATA_SIZE 4

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class PyramidLstmLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PyramidLstmLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_bottom2_(new Blob<Dtype>()),
        blob_bottom3_(new Blob<Dtype>()),
        blob_bottom4_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top2_(new Blob<Dtype>()),
        blob_top3_(new Blob<Dtype>()),
        blob_top4_(new Blob<Dtype>()){}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    vector<int> shape;
    shape.push_back(BATCH_SIZE);
    shape.push_back(INPUT_DATA_SIZE);
    shape.push_back(HEIGHT);
    shape.push_back(WIDTH);
    blob_bottom_->Reshape(shape);
    blob_bottom2_->Reshape(shape);
    blob_bottom3_->Reshape(shape);
    blob_bottom4_->Reshape(shape);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    GaussianFiller<Dtype> filler2(filler_param);
    GaussianFiller<Dtype> filler3(filler_param);
    GaussianFiller<Dtype> filler4(filler_param);
    
    filler.Fill(this->blob_bottom_);
    filler2.Fill(this->blob_bottom2_);
    filler3.Fill(this->blob_bottom3_);
    filler4.Fill(this->blob_bottom4_);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_bottom_vec_.push_back(blob_bottom3_);
    blob_bottom_vec_.push_back(blob_bottom4_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top2_);
    blob_top_vec_.push_back(blob_top3_);
    blob_top_vec_.push_back(blob_top4_);
  }
  virtual ~PyramidLstmLayerTest() {
    delete blob_bottom_;
    delete blob_bottom2_;
    delete blob_bottom3_;
    delete blob_top_;
    delete blob_top2_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_bottom3_;
  Blob<Dtype>* const blob_bottom4_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top2_;
  Blob<Dtype>* const blob_top3_;
  Blob<Dtype>* const blob_top4_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(PyramidLstmLayerTest, TestDtypesAndDevices);

TYPED_TEST(PyramidLstmLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PyramidLstmParameter* pyramid_lstm_param = layer_param.mutable_pyramid_lstm_param();
  pyramid_lstm_param->set_num_cells(NUM_CELLS);
  pyramid_lstm_param->mutable_weight_filler()->set_type("xavier");
  PyramidLstmLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for(int i = 0; i < this->blob_top_vec_.size(); i++)
  {
    Blob<Dtype>* blob_ = this->blob_top_vec_[i];
    EXPECT_EQ(blob_->num(), BATCH_SIZE * HEIGHT * WIDTH);
    EXPECT_EQ(blob_->channels(), NUM_CELLS);
    EXPECT_EQ(blob_->height(), 1);
    EXPECT_EQ(blob_->width(), 1);
  }
}

TYPED_TEST(PyramidLstmLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PyramidLstmParameter* pyramid_lstm_param = layer_param.mutable_pyramid_lstm_param();
  pyramid_lstm_param->set_num_cells(NUM_CELLS);
  pyramid_lstm_param->mutable_weight_filler()->set_type("xavier");
  PyramidLstmLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
}

TYPED_TEST(PyramidLstmLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PyramidLstmParameter* pyramid_lstm_param = layer_param.mutable_pyramid_lstm_param();
  pyramid_lstm_param->set_num_cells(NUM_CELLS);
  pyramid_lstm_param->mutable_weight_filler()->set_type("xavier");

  PyramidLstmLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //   this->blob_bottom_->mutable_cpu_diff()[i] = 1.;
  // }
  // for (int i = 0; i < this->blob_bottom2_->count(); ++i) {
  //   this->blob_bottom2_->mutable_cpu_diff()[i] = 1.;
  // }
  // vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  // layer.Backward(this->blob_top_vec_, propagate_down,
  //                this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, -1);
}

}  // namespace caffe
