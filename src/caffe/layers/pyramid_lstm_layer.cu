#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pyramid_lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void PyramidLstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}

INSTANTIATE_LAYER_GPU_FUNCS(PyramidLstmLayer);

}  // namespace caffe
