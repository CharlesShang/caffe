#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multiple_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultipleInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  vector<Blob<Dtype>*> blob_bottom_vec(1, NULL), blob_top_vec(1, NULL);
  blob_top_vec[0] = bottom[0];
  for (int i = 0; i < num_layer_ - 1; ++i)
  {
    blob_bottom_vec[0] = blob_top_vec[0];
    blob_top_vec[0]    = hidden_data_vec_[i].get();
    ip_layers_[i]->Forward(blob_bottom_vec, blob_top_vec);
    relu_layer_->Forward(blob_top_vec, blob_top_vec);
  }
  ip_layers_[num_layer_ - 1]->Forward(blob_top_vec, top);
  relu_layer_->Forward(top, top);

}

template <typename Dtype>
void MultipleInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  vector<Blob<Dtype>*> blob_top_vec(1, NULL), blob_bottom_vec(1, NULL);
  blob_bottom_vec[0] = top[0];
  for (int i = num_layer_ - 1; i > 0; --i){
    blob_top_vec[0] = blob_bottom_vec[0];
    blob_bottom_vec[0] = hidden_data_vec_[i-1].get();
    relu_layer_->Backward(blob_top_vec, propagate_down, blob_top_vec);
    ip_layers_[i]->Backward(blob_top_vec, propagate_down, blob_bottom_vec);
  }
  relu_layer_->Backward(blob_bottom_vec, propagate_down, blob_bottom_vec);
  ip_layers_[0]->Backward(blob_bottom_vec, propagate_down, bottom);

}

INSTANTIATE_LAYER_GPU_FUNCS(MultipleInnerProductLayer);

}  // namespace caffe
