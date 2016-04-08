#include <vector>
#include "boost/make_shared.hpp"
using boost::make_shared;
#include "caffe/filler.hpp"
#include "caffe/layers/pyramid_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void check_value_(Blob<Dtype> *in, Blob<Dtype> * out){
  for (int n = 0; n < in->num(); n++){
    for(int c = 0; c < in->channels(); c++) {
      for (int h = 0; h < in->height(); ++h){
        for(int w = 0; w < in->height(); w ++){
          int index1 = w + h * in->width() + c * in->height() * in->width()
            + n * in->channels() * in->height() * in->width();
          int index2 = (w + h * in->width()) * in->channels() + c
            + n * in->channels() * in->height() * in->width();
            if(fabs(in->cpu_data()[index1] - out->cpu_data()[index2]) > 1e-8){
                LOG(ERROR) << "data: "<< in->cpu_data()[index1] << " " << out->cpu_data()[index2] << "\n";
            }
        }
      }
    }
  }
}
template <typename Dtype>
void check_diff_(Blob<Dtype> *in, Blob<Dtype> * out){
  for (int n = 0; n < in->num(); n++){
    for(int c = 0; c < in->channels(); c++) {
      for (int h = 0; h < in->height(); ++h){
        for(int w = 0; w < in->height(); w ++){
          int index1 = w + h * in->width() + c * in->height() * in->width()
            + n * in->channels() * in->height() * in->width();
          int index2 = (w + h * in->width()) * in->channels() + c
            + n * in->channels() * in->height() * in->width();
            if(fabs(in->cpu_diff()[index1] - out->cpu_diff()[index2]) > 1e-8){
                LOG(ERROR) << "diff: " << in->cpu_diff()[index1] << " " << out->cpu_diff()[index2] << "\n";
            }
        }
      }
    }
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PyramidLstmParameter pyramid_lstm_param = this->layer_param_.pyramid_lstm_param();
  CHECK((pyramid_lstm_param.has_weight_filler()))
      << "pyramid_lstm_param.has_weight_filler()";
  CHECK((pyramid_lstm_param.has_num_cells()))
      << "pyramid_lstm_param.has_num_cells()";
  const int width  = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int batch  = bottom[0]->num();

  sequences_ = bottom.size();
  channels_ = pyramid_lstm_param.num_cells();
  num_ = batch * width * height;

  // call transpose layer setup
  LayerParameter transpose_param;
  transpose_layer_.reset(new TransposeLayer<Dtype>(transpose_param));
  transpose_bottom_vec_.clear();
  transpose_bottom_vec_.push_back(bottom[0]);
  transposed_data_.reset(new Blob<Dtype>());
  transposed_data_->Reshape(num_, channels, 1, 1);
  transpose_top_vec_.clear();
  transpose_top_vec_.push_back(transposed_data_.get());
  transpose_layer_->SetUp(transpose_bottom_vec_, transpose_top_vec_);

  // Setup lstm layer
  // add to learnable
  LayerParameter lstm_unit_param;
  lstm_unit_param.mutable_lstm_unit_param()->set_num_cells(channels_);
  lstm_unit_param.mutable_lstm_unit_param()->mutable_weight_filler()\
    ->CopyFrom(pyramid_lstm_param.weight_filler());
  lstm_layer_.reset(new LstmUnitLayer<Dtype>(lstm_unit_param));

  // NOTE: sequences + 1 intermediate states
  lstm_mem_cache_.clear();
  lstm_hidden_cache_.clear();
  for (int i = 0; i <= sequences_; ++i){
    lstm_mem_cache_.push_back(make_shared<Blob<Dtype> >(num_, channels_, 1, 1));  
    lstm_hidden_cache_.push_back(make_shared<Blob<Dtype> >(num_, channels_, 1, 1));
  }
  
  previous_hidden_ = lstm_hidden_cache_[0];
  previous_mem_    = lstm_mem_cache_[0];
  current_hidden_  = lstm_hidden_cache_[1];
  current_mem_     = lstm_mem_cache_[1];

  lstm_bottom_vec_.clear();
  lstm_top_vec_.clear();
  lstm_bottom_vec_.push_back(transposed_data_.get());
  lstm_bottom_vec_.push_back(previous_hidden_.get());
  lstm_bottom_vec_.push_back(previous_mem_.get());
  lstm_top_vec_.push_back(previous_hidden_.get());
  lstm_top_vec_.push_back(previous_mem_.get());
  lstm_layer_->SetUp(lstm_bottom_vec_, lstm_top_vec_);
  add_to_learnable(lstm_layer_->blobs(), this->blobs_);
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // CHECK((bottom.size() > 1))
  //     << "PyramidLstmLayer must have a sequential input ";
  CHECK_EQ(bottom.size(), top.size())
      << "PyramidLstmLayer top bottom are NOT equal";
  const int width  = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int batch  = bottom[0]->num();

  sequences_ = bottom.size();
  num_ = batch * width * height;
  transposed_data_->Reshape(num_, channels, 1, 1);
  transpose_bottom_vec_.resize(1);
  transpose_top_vec_.resize(1);
  for (int i = 0; i < sequences_; i ++)
  {
    top[i]->Reshape(num_, channels_, 1, 1);
  }
  for (int i = 0; i <= sequences_; ++i){
    lstm_mem_cache_[i]->Reshape(num_, channels_, 1, 1);  
    lstm_hidden_cache_[i]->Reshape(num_, channels_, 1, 1);
  }
  previous_hidden_ = lstm_hidden_cache_[0];
  previous_mem_    = lstm_mem_cache_[0];
  current_hidden_  = lstm_hidden_cache_[1];
  current_mem_     = lstm_mem_cache_[1];
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(sequences_, bottom.size());
  CHECK_EQ(sequences_, top.size());
  CHECK_EQ(previous_hidden_, lstm_hidden_cache_[0]) 
    << "Forward propagate starts at the wrong place";
  CHECK_EQ(current_hidden_, lstm_hidden_cache_[1]) 
    << "Forward propagate starts at the wrong place";
  lstm_bottom_vec_.resize(3);
  lstm_bottom_vec_[0] = transposed_data_.get(); 
  lstm_bottom_vec_[1] = previous_hidden_.get();
  lstm_bottom_vec_[2] = previous_mem_.get();
  // set the seed to zeros
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), lstm_bottom_vec_[1]->mutable_cpu_data());
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), lstm_bottom_vec_[2]->mutable_cpu_data());
  lstm_top_vec_.resize(2);
  lstm_top_vec_[0] = current_hidden_.get();
  lstm_top_vec_[1] = current_mem_.get();

  for (int i = 0; i < sequences_; ++i){
    CHECK_EQ(top[i]->count(), num_ * channels_);
    Blob<Dtype> * input = bottom[i];
    Blob<Dtype> * output = top[i];
    // N*C*H*W -> (N*H*W)*C*1*1
    // TODO: avoid transpose
    transpose_blob_forward(input, lstm_bottom_vec_[0]);
    // FP lstm 
    lstm_layer_->Forward(lstm_bottom_vec_, lstm_top_vec_);
    // copy to output
    caffe_copy<Dtype>(num_ * channels_, lstm_top_vec_[0]->cpu_data(), 
      output->mutable_cpu_data());
    // prepare next computing
    if ( i < sequences_ - 1)
    {
      previous_hidden_ = lstm_hidden_cache_[i+1];
      current_hidden_  = lstm_hidden_cache_[i+2];
      previous_mem_    = lstm_mem_cache_[i+1];
      current_mem_     = lstm_mem_cache_[i+2];
      lstm_bottom_vec_[1] = previous_hidden_.get();
      lstm_bottom_vec_[2] = previous_mem_.get();
      lstm_top_vec_[0] = current_hidden_.get();
      lstm_top_vec_[1] = current_mem_.get();
    }
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(current_hidden_, lstm_hidden_cache_[sequences_]) 
    << "Have not forward propagate the whole sequence";
  CHECK_EQ(previous_hidden_, lstm_hidden_cache_[sequences_-1]) 
    << "Have not forward propagate the whole sequence";
  CHECK_EQ(current_mem_, lstm_mem_cache_[sequences_]) 
    << "Have not forward propagate the whole sequence";
  CHECK_EQ(previous_mem_, lstm_mem_cache_[sequences_-1]) 
    << "Have not forward propagate the whole sequence";
  // set diff to zero
  lstm_bottom_vec_.resize(3);
  lstm_bottom_vec_[0] = transposed_data_.get(); 
  lstm_bottom_vec_[1] = previous_hidden_.get();
  lstm_bottom_vec_[2] = previous_mem_.get();
  lstm_top_vec_.resize(2);
  lstm_top_vec_[0] = current_hidden_.get();
  lstm_top_vec_[1] = current_mem_.get();
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), current_hidden_->mutable_cpu_diff());
  caffe_set<Dtype>(num_ * channels_, Dtype(0.), current_mem_->mutable_cpu_diff());
  for (int i = sequences_ - 1; i >= 0; --i){
    Blob<Dtype> * output = top[i];
    Blob<Dtype> * input = bottom[i];
    // add the two branches diff 
    caffe_cpu_axpby<Dtype>(num_ * channels_, Dtype(1), output->cpu_diff(),
      Dtype(1), lstm_top_vec_[0]->mutable_cpu_diff());
    // BP lstm
    transpose_blob_forward(input, lstm_bottom_vec_[0]);
    lstm_layer_->Forward(lstm_bottom_vec_, lstm_top_vec_);
    vector<bool> propagate_down(lstm_bottom_vec_.size(), true);
    lstm_layer_->Backward(lstm_top_vec_, propagate_down, lstm_bottom_vec_);
    // BP transpose
    transpose_blob_backward(lstm_bottom_vec_[0], input);
    // prepare next computing
    if (i > 0){
      previous_hidden_ = lstm_hidden_cache_[i-1];
      current_hidden_  = lstm_hidden_cache_[i];
      previous_mem_    = lstm_mem_cache_[i-1];
      current_mem_     = lstm_mem_cache_[i];
      lstm_bottom_vec_[1] = transposed_data_.get();
      lstm_bottom_vec_[1] = previous_hidden_.get();
      lstm_bottom_vec_[2] = previous_mem_.get();
      lstm_top_vec_[0] = current_hidden_.get();
      lstm_top_vec_[1] = current_mem_.get();
    }
  }
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::transpose_blob_forward(Blob<Dtype> * bottom, 
    Blob<Dtype> * top){
  transpose_bottom_vec_[0] = bottom;
  transpose_top_vec_[0] = top;
  transpose_layer_->Forward(transpose_bottom_vec_, transpose_top_vec_);
}
template <typename Dtype>
void PyramidLstmLayer<Dtype>::transpose_blob_backward(Blob<Dtype> * top, 
    Blob<Dtype> * bottom){
  transpose_bottom_vec_[0] = bottom;
  transpose_top_vec_[0] = top;
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
