#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pyramid_lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reverse_transpose_blob_forward_gpu(
  int count,
  int num, int channels, int height, int width,
  const Dtype * from,
  Dtype * to ){
  const int sz1 = height * width;
  const int sz2 = channels * sz1;
  const int sz3 = width * channels;
  CUDA_KERNEL_LOOP(idx, count){
    int w = idx % width;
    int h = idx / width % height;
    int c = idx / sz1 % channels;
    int n = idx / sz2;
    const int idx2 = w * channels + h * sz3 + n * sz2 + c;
    to[idx] = from[idx2];
  }
}
template <typename Dtype>
__global__ void reverse_transpose_blob_backward_gpu(
  int count,
  int num, int channels, int height, int width,
  const Dtype * from,
  Dtype * to){
  const int sz1 = height * width;
  const int sz2 = channels * sz1;
  const int sz3 = width * channels;
  CUDA_KERNEL_LOOP(idx, count){
    int w = idx % width;
    int h = idx / width % height;
    int c = idx / sz1 % channels;
    int n = idx / sz2;
    const int idx2 = w * channels + h * sz3 + n * sz2 + c;
    to[idx2] = from[idx];
  }
}


template <typename Dtype>
void PyramidLstmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

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
    Blob<Dtype> * input = bottom[i];
    Blob<Dtype> * output = top[i];
    // N*C*H*W -> (N*H*W)*C*1*1
    // TODO: avoid transpose
    transpose_blob_forward(input, lstm_bottom_vec_[0]);
    // FP lstm 
    lstm_layers_[i]->Forward(lstm_bottom_vec_, lstm_top_vec_);
    // copy to output
    reverse_transpose_blob_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(num_ * channels_),
        CAFFE_CUDA_NUM_THREADS>>>
      (output->count(), output->num(), output->channels(), output->height(), output->width(),
      lstm_top_vec_[0]->gpu_data(), output->mutable_gpu_data());
    // prepare next computing
    if ( i < sequences_ - 1){
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
void PyramidLstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
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
  shared_ptr<Blob<Dtype> > top_transposed_tmp;
  top_transposed_tmp.reset(new Blob<Dtype>(num_, channels_, 1, 1));
  for (int i = sequences_ - 1; i >= 0; --i){
    Blob<Dtype> * output = top[i];
    Blob<Dtype> * input = bottom[i];
    // add two branches diff 
    reverse_transpose_blob_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(num_ * channels_),
        CAFFE_CUDA_NUM_THREADS>>>
      (output->count(),output->num(), output->channels(), output->height(), output->width(),
      output->gpu_diff(), top_transposed_tmp->mutable_gpu_diff());
    caffe_gpu_axpby<Dtype>(num_ * channels_, Dtype(1), top_transposed_tmp->gpu_diff(),
      Dtype(1), lstm_top_vec_[0]->mutable_gpu_diff());
    // BP lstm
    transpose_blob_forward(input, lstm_bottom_vec_[0]);
    vector<bool> propagate_down(lstm_bottom_vec_.size(), true);

    lstm_layers_[i]->Backward(lstm_top_vec_, propagate_down, lstm_bottom_vec_);
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
INSTANTIATE_LAYER_GPU_FUNCS(PyramidLstmLayer);
}  // namespace caffe
