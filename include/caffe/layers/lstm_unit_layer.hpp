#ifndef CAFFE_LSTM_UNIT_LAYER_HPP_
#define CAFFE_LSTM_UNIT_LAYER_HPP_

#include <vector>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/concat_layer.hpp"

namespace caffe {

template <typename Dtype>
class LstmUnitLayer : public Layer<Dtype> {
 public:
  explicit LstmUnitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool overwrites_param_diffs() { return false; }
  virtual inline const char* type() const { return "LstmUnitLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;  // num memory cells;
  int num_;  // batch size;
  int input_data_size_;
  int M_;
  int N_;
  int K_;
  vector<shared_ptr<Blob<Dtype> > > buffers_;
  shared_ptr<Blob<Dtype> > input_gates_data_buffer_;
  shared_ptr<Blob<Dtype> > forget_gates_data_buffer_;
  shared_ptr<Blob<Dtype> > output_gates_data_buffer_;
  shared_ptr<Blob<Dtype> > input_values_data_buffer_;
  shared_ptr<Blob<Dtype> > gates_diff_buffer_;
  shared_ptr<Blob<Dtype> > next_state_tot_diff_buffer_;
  shared_ptr<Blob<Dtype> > tanh_mem_buffer_;
  shared_ptr<Blob<Dtype> > dldg_buffer_;

  // the internal concat layer for concat the input and previous state
  shared_ptr<ConcatLayer<Dtype> > concat_layer_;
  // the data concated using input and previous memory cell
  shared_ptr<Blob<Dtype> > concated_data_;
  vector<Blob<Dtype>*> concat_bottom_vec_;
  vector<Blob<Dtype>*> concat_top_vec_;
};

}
#endif
