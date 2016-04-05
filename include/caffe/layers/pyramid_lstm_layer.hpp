#ifndef CAFFE_PYRAMID_LSTM_LAYER_HPP_
#define CAFFE_PYRAMID_LSTM_LAYER_HPP_

#include <vector>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/lstm_unit_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/transpose_layer.hpp"

namespace caffe {
/**
 * @brief Does pyramid lstm on the input deep features
 */
template <typename Dtype>
class PyramidLstmLayer : public Layer<Dtype> {
 public:
  explicit PyramidLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PyraimdLstmLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void add_to_learnable(vector<shared_ptr<Blob<Dtype> > > &lstm_blobs,
      vector<shared_ptr<Blob<Dtype> > > &this_blobs){
    this_blobs.clear();
    for (int i = 0; i < lstm_blobs.size(); i ++){
      this_blobs.push_back(lstm_blobs[i]);
    }
  }
  void transpose_blob_forward( Blob<Dtype> * bottom, Blob<Dtype> * top); // to transposed_data_
  void transpose_blob_backward( Blob<Dtype> * top, Blob<Dtype> * bottom); // to transposed_data_

  int channels_;  // memory cells;
  int num_;       // batch size;
  int sequences_;  // number of sequences

  // lstm unit
  shared_ptr<LstmUnitLayer<Dtype> > lstm_layer_;
  vector<Blob<Dtype> *> lstm_bottom_vec_;
  vector<Blob<Dtype> *> lstm_top_vec_;
  shared_ptr<Blob<Dtype> > previous_hidden_; //should not share data with the output
  shared_ptr<Blob<Dtype> > previous_mem_;
  // inplace?
  shared_ptr<Blob<Dtype> > current_hidden_;  //should not share data with the output
  shared_ptr<Blob<Dtype> > current_mem_;

  // transpose N*C*H*W blob into (N*H*W)*C*1*1 blob
  // before the lstm unit
  shared_ptr<TransposeLayer<Dtype> > transpose_layer_;
  vector<Blob<Dtype> *> transpose_bottom_vec_;
  vector<Blob<Dtype> *> transpose_top_vec_;
  shared_ptr<Blob<Dtype> > transposed_data_;
  // there should be reverse transpose layer
  // TODO
};

}
#endif
