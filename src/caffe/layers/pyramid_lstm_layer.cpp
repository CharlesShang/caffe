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
  for (int i = 0; i < sequences_; i ++)
  {
    top[i]->Reshape(num_, channels_, 1, 1);
  }
  previous_hidden_->Reshape(num_, channels_, 1, 1);
  previous_mem_->Reshape(num_, channels_, 1, 1);
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(sequences_, bottom.size());
  CHECK_EQ(sequences_, top.size());
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), previous_mem_->mutable_cpu_data());
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), previous_hidden_->mutable_cpu_data());
  lstm_bottom_vec_.clear();
  lstm_bottom_vec_.push_back(transposed_data_.get());
  lstm_bottom_vec_.push_back(previous_hidden_.get());
  lstm_bottom_vec_.push_back(previous_mem_.get());
  lstm_top_vec_.clear();
  lstm_top_vec_.push_back(previous_hidden_.get());
  lstm_top_vec_.push_back(previous_mem_.get());
  for (int i = 0; i < sequences_; ++i){
    CHECK_EQ(top[i]->count(), num_ * channels_);
    Blob<Dtype> * input = bottom[i];
    Blob<Dtype> * output = top[i];
    // N*C*H*W -> (N*H*W)*C*1*1
    transpose_blob_forward(input, transposed_data_.get());
    lstm_bottom_vec_[0] = transposed_data_.get();
    // FP lstm 
    lstm_layer_->Forward(lstm_bottom_vec_, lstm_top_vec_);
    // copy to output
    caffe_copy<Dtype>(num_ * channels_, previous_hidden_->cpu_data(), 
      output->mutable_cpu_data());
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // set diff to zero
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), previous_mem_->mutable_cpu_diff());
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), previous_hidden_->mutable_cpu_diff());
  lstm_bottom_vec_.clear();
  lstm_bottom_vec_.push_back(transposed_data_.get());
  lstm_bottom_vec_.push_back(previous_hidden_.get());
  lstm_bottom_vec_.push_back(previous_mem_.get());
  lstm_top_vec_.clear();
  lstm_top_vec_.push_back(previous_hidden_.get());
  lstm_top_vec_.push_back(previous_mem_.get());
  for (int i = sequences_ - 1; i >= 0; --i){
    Blob<Dtype> * output = top[i];
    Blob<Dtype> * input = bottom[i];
    // add the current diff and the diff from next state
    caffe_add<Dtype>(num_ * channels_, previous_hidden_->cpu_diff(),
      output->cpu_diff(), previous_hidden_->mutable_cpu_diff());
    // BP lstm
    vector<bool> propagate_down(3, true);
    lstm_layer_->Backward(lstm_top_vec_, propagate_down, lstm_bottom_vec_);
    // BP transpose
    transpose_blob_backward(transposed_data_.get(), input);
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::transpose_blob_forward(Blob<Dtype> * bottom, 
    Blob<Dtype> * top){
  transpose_bottom_vec_.clear();
  transpose_top_vec_.clear();
  transpose_bottom_vec_.push_back(bottom);
  transpose_top_vec_.push_back(top);
  transpose_layer_->Forward(transpose_bottom_vec_, transpose_top_vec_);
}
template <typename Dtype>
void PyramidLstmLayer<Dtype>::transpose_blob_backward(Blob<Dtype> * top, 
    Blob<Dtype> * bottom){
  transpose_bottom_vec_.clear();
  transpose_top_vec_.clear();
  transpose_bottom_vec_.push_back(bottom);
  transpose_top_vec_.push_back(top);
  vector<bool> propagate_down(1, true);
  transpose_layer_->Backward(transpose_top_vec_, propagate_down, 
    transpose_bottom_vec_);
}

#ifdef CPU_ONLY
STUB_GPU(PyramidLstmLayer);
#endif

INSTANTIATE_CLASS(PyramidLstmLayer);
REGISTER_LAYER_CLASS(PyramidLstm);

}  // namespace caffe
