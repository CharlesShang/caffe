#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
inline void debug_print(Blob<Dtype> *x) {
  std::cout << "axis" << x->num_axes() << "\n";
  const Dtype * data = x->cpu_data();
  for (int n = 0; n < x->num(); n++){
    std::cout << "num " << n << "\n";
    for(int c = 0; c < x->channels(); c++) {
      std::cout << "channel " << c << "\n";
      for (int h = 0; h < x->height(); ++h){
        for(int w = 0; w < x->height(); w ++){
          int index = w + h * w + c * h * w + n * c * h * w;
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

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num() * bottom[0]->height() * bottom[0]->width();
  const int channel = bottom[0]->channels();
  top[0]->Reshape(num, channel, 1, 1);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int offset = bottom[0]->count() / bottom[0]->num() * n;
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < spatial_dim; ++i) {
        top_data[offset + i * channels + c]
            = bottom_data[offset + c * spatial_dim + i];
      }
    }
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int offset = bottom[0]->count() / bottom[0]->num() * n;
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < spatial_dim; ++i) {
        bottom_diff[offset + c * spatial_dim + i] =
            top_diff[offset + i * channels + c];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
