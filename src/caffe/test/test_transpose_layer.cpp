#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/transpose_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 3
#define WIDTH  3
#define HEIGHT 2
#define INPUT_DATA_SIZE 2

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class TransposeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TransposeLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()){}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    vector<int> shape;
    shape.push_back(BATCH_SIZE);
    shape.push_back(INPUT_DATA_SIZE);
    shape.push_back(HEIGHT);
    shape.push_back(WIDTH);
    blob_bottom_->Reshape(shape);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);


    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

  }
  virtual ~TransposeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename Dtype>
void check_value(Blob<Dtype> *in, Blob<Dtype> * out){
    for (int n = 0; n < in->num(); n++){
      for(int c = 0; c < in->channels(); c++) {
        for (int h = 0; h < in->height(); ++h){
          for(int w = 0; w < in->height(); w ++){
            int index1 = w + h * in->width() + c * in->height() * in->width()
              + n * in->channels() * in->height() * in->width();
            int index2 = (w + h * in->width()) * in->channels() + c
              + n * in->channels() * in->height() * in->width();
              EXPECT_NEAR(in->cpu_data()[index1], out->cpu_data()[index2], 1e-8);
              EXPECT_NEAR(in->cpu_diff()[index1], out->cpu_diff()[index2], 1e-8);
          }
        }
      }
    }
  }

template <typename Dtype>
inline void debug_print(Blob<Dtype> *x) {
  std::cout << "axis" << x->num_axes() << "\n";
  const Dtype * data = x->cpu_data();
  for (int n = 0; n < x->num(); n++){
    std::cout << "num " << n << "\n";
    for(int c = 0; c < x->channels(); c++) {
      std::cout << "channel " << c << "\n";
      for (int h = 0; h < x->height(); ++h){
        for(int w = 0; w < x->width(); w ++){
          int index = w + h * x->width() + c * x->height() * x->width()
           + n * x->channels() * x->height() * x->width();
          std::cout << data[index] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

TYPED_TEST_CASE(TransposeLayerTest, TestDtypesAndDevices);

TYPED_TEST(TransposeLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransposeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for(int i = 0; i < this->blob_top_vec_.size(); i++)
  {
    Blob<Dtype>* blob_ = this->blob_top_vec_[i];
    EXPECT_EQ(blob_->num(), BATCH_SIZE * HEIGHT * WIDTH);
    EXPECT_EQ(blob_->channels(), INPUT_DATA_SIZE);
    EXPECT_EQ(blob_->height(), 1);
    EXPECT_EQ(blob_->width(), 1);
  }
}

TYPED_TEST(TransposeLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransposeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // debug_print(this->blob_bottom_vec_[0]);
  // debug_print(this->blob_top_vec_[0]);
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  check_value(this->blob_top_vec_[0], this->blob_top_vec_[0]);
  // debug_print(this->blob_bottom_vec_[0]);
  // debug_print(this->blob_top_vec_[0]);
}

TYPED_TEST(TransposeLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransposeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-4);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // for (int i = 0; i < this->blob_top_->count(); ++i) {
  //   this->blob_top_->mutable_cpu_diff()[i] = 1.;
  // }
  // vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  // layer.Backward(this->blob_top_vec_, propagate_down,
  //                this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
