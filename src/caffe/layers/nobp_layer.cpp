#include <algorithm>
#include <vector>

#include "caffe/layers/nobp_layer.hpp"

namespace caffe {

template <typename Dtype>
void NoBPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    // just copy 
    top_data[i] = bottom_data[i];
  }
}

template <typename Dtype>
void NoBPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      // no bp error 
      bottom_diff[i] = 0;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NoBPLayer);
#endif

INSTANTIATE_CLASS(NoBPLayer);

}  // namespace caffe
