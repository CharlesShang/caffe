#ifndef CAFFE_CLUSTERING_LAYER_HPP_
#define CAFFE_CLUSTERING_LAYER_HPP_

#include <vector>
#include <utility> 

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class ClusteringLayer : public Layer<Dtype> {
 public:
  explicit ClusteringLayer(const LayerParameter& param)
      : Layer<Dtype>(param), param_seted_(false){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);      
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Clustering"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  int k_;  // k centers
  int total_class_; // total classes for the finnal classification problem
  bool param_seted_;
  Dtype lambda_;  // loss weight
  bool branch_;
  bool across_class_;
  
  // total_class * k clusters, each shape 1 * c * h * w
  vector<vector<shared_ptr<Blob<Dtype> > > > centroids_; 

  // ip layers
  int num_output_;
  vector<vector<shared_ptr<InnerProductLayer<Dtype> > > > ip_layer_vec_; // each cluster should has a distict transform
  vector<Blob<Dtype> *>  ip_top_vec_; // 
  vector<Blob<Dtype> *>  ip_bottom_vec_; // 

  // training
  vector<pair<int, int> > assigned_centers_;  // record the center for BP,
  vector<Dtype> clustering_loss_;             // Sw/Sb
  vector<Dtype> dist_;

  // cache data during BP
  // perform kmeans after caching enough data
#define CLUSTERING_CACHE_DATA_SIZE_MAX_  10000
  vector<int> cache_data_size_;           // total_class * cache_data_size
  vector<shared_ptr<Blob<Dtype> > > cache_data_; 
  vector<shared_ptr<Blob<Dtype> > > cache_label_;

  // helper for build this layer
  void add_to_learnable(vector<shared_ptr<Blob<Dtype> > > &blobs,
    vector<shared_ptr<Blob<Dtype> > > &this_blobs){
    for (int i = 0; i < blobs.size(); i ++){
      this_blobs.push_back(blobs[i]);
    }
  }
  void setup_ip_layers(int channels, int height, int width);

  // kmeans functions
  void kmeans(const Dtype * data, int n, int m, int k, 
    vector<shared_ptr<Blob<Dtype> > > & centroids,
    Dtype * labels);
  int nearest(const Dtype * data, int m, int k, 
    vector<shared_ptr<Blob<Dtype> > > & centroids, Dtype & dist);
  double cal_dist(int m, const Dtype * x, const Dtype * y, Dtype *tmp){
    // WRONG!!!
    // caffe_sub<Dtype>(m, x, y, tmp);
    // caffe_mul<Dtype>(m, tmp, tmp, tmp);
    // return caffe_cpu_dot<Dtype>(m, tmp, tmp) * 0.5;
    double dist = 0;
    for (int i = 0; i < m; ++i){
      dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return dist * 0.5;
  }
  double cal_dist(int m, const Dtype * x, const Dtype * y, Blob<Dtype>* tmp){
    caffe_sub<Dtype>(m, x, y, tmp->mutable_cpu_data());
    return tmp->sumsq_data() * 0.5;
  }
  // kmeans ++ init
  void kmpp(const Dtype * data, int n, int m, int k, 
    vector<shared_ptr<Blob<Dtype> > > & centroids);
  inline string vector_to_string(vector<int> &v) const {
    ostringstream stream;
    int s = 0;
    for (int i = 0; i < v.size(); ++i) {
      stream << v[i] << " ";
      s += v[i];
    }
    stream << "(" << s << ")";
    return stream.str();
  }
  inline string blob_to_string(const Dtype * data, int m) const {
    ostringstream stream;
    for (int i = 0; i < m; ++i) {
      stream << data[i] << " ";
    }
    return stream.str();
  }
};

}  // namespace caffe

#endif  // CAFFE_CLUSTERING_LAYER_HPP_
