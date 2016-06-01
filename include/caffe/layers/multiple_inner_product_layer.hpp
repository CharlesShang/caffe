#ifndef CAFFE_MULTIPLE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_MULTIPLE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
using google::protobuf::RepeatedPtrField;

namespace caffe {

template <typename Dtype>
class MultipleInnerProductLayer : public Layer<Dtype> {
 public:
  explicit MultipleInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param), param_seted_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultipleInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<shared_ptr<InnerProductLayer<Dtype> > > ip_layers_;
  shared_ptr<ReLULayer<Dtype> > relu_layer_;
  int num_layer_;
  vector<int> num_outputs_;
  // hidden_data_vec_.size() == num_layer_ - 1
  vector<shared_ptr<Blob<Dtype> > > hidden_data_vec_;
  // only for forward IP layers
  vector<Blob<Dtype> *> ip_bottom_;
  vector<Blob<Dtype> *> ip_top_;

  // when loading weights from a file, the inner product layers's weight must be setted
  bool param_seted_;

  void add_to_learnable(vector<shared_ptr<Blob<Dtype> > > &blobs,
    vector<shared_ptr<Blob<Dtype> > > &this_blobs){
    for (int i = 0; i < blobs.size(); i ++){
      this_blobs.push_back(blobs[i]);
    }
  }
  void setup_ip_layer(int bottom_dim, int top_dim, 
  shared_ptr<InnerProductLayer<Dtype> > & ip_layer);
};

}  // namespace caffe

#endif  // CAFFE_MULTIPLE_INNER_PRODUCT_LAYER_HPP_
