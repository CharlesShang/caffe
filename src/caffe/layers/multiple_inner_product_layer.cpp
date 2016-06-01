#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multiple_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultipleInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  param_seted_ = true;
  num_layer_ = this->layer_param_.multiple_inner_product_param().num_layer();
  string activation = this->layer_param_.multiple_inner_product_param().activation();
  CHECK_EQ(num_layer_, this->layer_param_.multiple_inner_product_param().num_outputs_size())
    << "Multiple Inner Product Layer Setup Wrong";
  num_outputs_.resize(num_layer_);
  for (int i = 0; i < num_outputs_.size(); ++i){
    num_outputs_[i] = this->layer_param_.multiple_inner_product_param().num_outputs(i);
  }
  const int channels = bottom[0]->channels();
  const int height   = bottom[0]->height();
  const int width    = bottom[0]->width();

  // set up iplayers
  int bottom_dim, top_dim;
  top_dim = channels * height * width;
  ip_layers_.resize(num_layer_);
  for (int i = 0; i < num_layer_; ++i){
    bottom_dim = top_dim;
    top_dim = num_outputs_[i];
    setup_ip_layer(bottom_dim, top_dim, ip_layers_[i]);
    // add to learnable
    add_to_learnable(ip_layers_[i]->blobs(), this->blobs_);
  }
  if (num_layer_ >= 2){
    hidden_data_vec_.resize(num_layer_ - 1);
  }else{
    hidden_data_vec_.clear();
  }

  // set up relu layer
  LayerParameter relu_param;
  relu_layer_.reset(new ReLULayer<Dtype>(relu_param));

}

template <typename Dtype>
void MultipleInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height   = bottom[0]->height();
  const int width    = bottom[0]->width();

  // LOG(ERROR) << "Check point 1" ;
  if (! param_seted_){
    // set up ip by copying weights from this blobs
    param_seted_ = true;
    num_layer_ = this->blobs_.size() / 2;
    num_outputs_.resize(num_layer_);
    ip_layers_.resize(num_layer_);
    CHECK_EQ(num_layer_, this->layer_param_.multiple_inner_product_param().num_layer())
      << "layers' number do not match";
    // LOG(ERROR) << "Check point 2, layer number " << num_layer_;
    int bottom_dim, top_dim;
    top_dim = channels * height * width;
    if (num_layer_ >= 2){
      hidden_data_vec_.resize(num_layer_ - 1);
    }else{
      hidden_data_vec_.clear();
    }
    // LOG(ERROR) << "Check point 3, this blobs_ " << this->blobs_.size();
    for (int i = 0; i < num_layer_; ++i){
      num_outputs_[i] = this->blobs_[i * 2]->shape()[0];
      bottom_dim = top_dim;
      top_dim = num_outputs_[i];      
      setup_ip_layer(bottom_dim, top_dim, ip_layers_[i]);
      ip_layers_[i]->blobs()[0] = this->blobs_[i*2];
      ip_layers_[i]->blobs()[1] = this->blobs_[i*2 + 1];
    }
    // LOG(ERROR) << "Check point 4, top dim " << num_outputs_[num_layer_ - 1];
    // relu layer
    LayerParameter relu_param;
    relu_layer_.reset(new ReLULayer<Dtype>(relu_param));
  }
  vector<int> shape(2, 0);
  shape[0] = num;
  for (int i = 0; i < num_layer_ - 1; ++i){
    shape[1] = num_outputs_[i];
    hidden_data_vec_[i].reset(new Blob<Dtype>(shape));
  }
  shape[1] = num_outputs_[num_layer_ - 1]; 
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MultipleInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void MultipleInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

template <typename Dtype>
void MultipleInnerProductLayer<Dtype>::setup_ip_layer(int bottom_dim, int top_dim, 
  shared_ptr<InnerProductLayer<Dtype> > & ip_layer){

  LayerParameter inner_product_param;
  inner_product_param.mutable_inner_product_param()->set_num_output(top_dim);
  inner_product_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
  inner_product_param.mutable_inner_product_param()->set_bias_term(true);

  vector<int> shape_bottom(2, 0), shape_top(2, 0);
  shape_top[0] = 1;    shape_top[1] = top_dim;  
  shape_bottom[0] = 1; shape_bottom[1] = bottom_dim;
  shared_ptr<Blob<Dtype> > blob_bottom, blob_top;
  blob_bottom.reset(new Blob<Dtype>(shape_bottom));
  blob_top.reset(new Blob<Dtype>(shape_top));
  vector<Blob<Dtype> *> blob_bottom_vec, blob_top_vec;
  blob_bottom_vec.push_back(blob_bottom.get());
  blob_top_vec.push_back(blob_top.get());

  ip_layer.reset(new InnerProductLayer<Dtype>(inner_product_param));
  ip_layer->SetUp(blob_bottom_vec, blob_top_vec);
}

#ifdef CPU_ONLY
STUB_GPU(MultipleInnerProductLayer);
#endif

INSTANTIATE_CLASS(MultipleInnerProductLayer);
REGISTER_LAYER_CLASS(MultipleInnerProduct);

}  // namespace caffe
