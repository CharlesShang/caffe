#include <vector>

#include "caffe/layers/absdist_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsdistLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype abs;
  caffe_gpu_asum(count, diff_.gpu_data(), &abs);
  Dtype loss = abs / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
  for(int i = 0; i < count; i++)
  {
    diff_.mutable_cpu_data()[i] = diff_.cpu_data()[i] > 0.0 ? 1.0 : -1.0;
  }
}

template <typename Dtype>
void AbsdistLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsdistLossLayer);

}  // namespace caffe
