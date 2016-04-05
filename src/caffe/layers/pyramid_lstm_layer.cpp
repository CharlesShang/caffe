#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/pyramid_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PyramidLstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PyramidLstmParameter pyramid_lstm_param = this->layer_param_.pyramid_lstm_param();
  CHECK((pyramid_lstm_param.has_weight_filler()))
      << "pyramid_lstm_param.has_weight_filler()";
  CHECK((pyramid_lstm_param.has_num_cells()))
      << "pyramid_lstm_param.has_num_cells()";
  const int width  = bottom[0]->shape(3);
  const int height = bottom[0]->shape(2);
  const int channel= bottom[0]->shape(1);
  const int batch  = bottom[0]->shape(0);

  sequences_ = bottom.size();
  channels_ = pyramid_lstm_param.num_cells();
  num_ = batch * width * height;

  // call transpose layer setup
  LayerParameter transpose_param;
  transpose_layer_.reset(new TransposeLayer<Dtype>(transpose_param));
  transpose_bottom_vec_.push_back(bottom[0]);
  transposed_data_.reset(new Blob<Dtype>());
  transpose_top_vec_.push_back(transposed_data_.get());
  transpose_layer_->SetUp(transpose_bottom_vec_, transpose_top_vec_);

  // call lstm_layer setup
  // add to learnable
  LayerParameter lstm_unit_param;
  lstm_unit_param.mutable_lstm_unit_param()->set_num_cells(channels_);
  lstm_unit_param.mutable_lstm_unit_param()->mutable_weight_filler()\
    ->CopyFrom(pyramid_lstm_param.weight_filler());
  lstm_layer_.reset(new LstmUnitLayer<Dtype>(lstm_unit_param));
  previous_hidden_.reset(new Blob<Dtype>());
  previous_mem_.reset(new Blob<Dtype>());
  lstm_bottom_vec_.clear();
  lstm_top_vec_.clear();
  lstm_top_vec_.push_back(previous_hidden_.get());
  lstm_top_vec_.push_back(previous_mem_.get());
  lstm_bottom_vec_.push_back(transposed_data_.get());
  lstm_bottom_vec_.push_back(previous_hidden_.get());
  lstm_bottom_vec_.push_back(previous_mem_.get());
  lstm_layer_->SetUp(lstm_bottom_vec_, lstm_top_vec_);
  add_to_learnable(lstm_layer_->blobs(), this->blobs_);
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK((this->layer_param_.bottom_size() > 1))
      << "PyramidLstmLayer must have a sequential input";
  CHECK_EQ(this->layer_param_.bottom_size(), \
            this->layer_param_.top_size())
      << "PyramidLstmLayer top bottom are NOT equal";
  const int width  = bottom[0]->shape(3);
  const int height = bottom[0]->shape(2);
  const int batch  = bottom[0]->shape(0);

  sequences_ = bottom.size();
  num_ = batch * width * height;

  vector<int> shape;
  shape.push_back(num_);
  shape.push_back(channels_);
  for (int i = 0; i < sequences_; i ++)
  {
    top[i]->Reshape(batch, channels_, height, width);
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(sequences_, bottom.size());
  CHECK_EQ(sequences_, top.size());
  for (int i = 0; i < sequences_; ++i){
    transpose_cpu_blob(bottom[i]);
    
    // FP lstm first
    // copy hidden to top
  }
  // reshape to meet the top size
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // set previous_mem_ diff to zero
  for (int i = 0; i < sequences_; ++i){
    // add top's diff to previous diff
    // BP each lstm
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::transpose_cpu_blob( Blob<Dtype> * blob){
  transpose_bottom_vec_.clear();
  transpose_top_vec_.clear();
  transpose_bottom_vec_.push_back(blob);
  transpose_top_vec_.push_back(transposed_data_.get());
  transpose_layer_->Forward(transpose_bottom_vec_, transpose_top_vec_);
}

#ifdef CPU_ONLY
STUB_GPU(PyramidLstmLayer);
#endif

INSTANTIATE_CLASS(PyramidLstmLayer);
REGISTER_LAYER_CLASS(PyramidLstm);

}  // namespace caffe
