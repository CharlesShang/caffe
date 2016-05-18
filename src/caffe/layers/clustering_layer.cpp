#include <vector>
#include <utility>
#include "boost/make_shared.hpp"
using boost::make_shared;
#include "caffe/filler.hpp"
#include "caffe/layers/clustering_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClusteringLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ClusteringParameter clustering_param = this->layer_param_.clustering_param();
  k_ = clustering_param.k();
  total_class_ = clustering_param.total_class();
  param_seted_ = true;
  lambda_ = clustering_param.lambda();  //current layers' loss weight
  branch_ = clustering_param.branch();
  across_class_ = clustering_param.across_class();
  total_class_  = across_class_ ? 1 : clustering_param.total_class();
  // CHECK_EQ(bottom.size()==1 || bottom.size()==2, total_class_ == 1)
  //     << "general clustring ";
  // CHECK_EQ(bottom.size()==2, total_class_ == 2) //TODO: make this dynamic
  //     << "inter class clustring ";

  const int width  = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  // const int num  = bottom[0]->num();
  if(clustering_param.has_num_output())
      num_output_ = clustering_param.num_output();
  else
      num_output_ = channels * height * width;

  // setup innerproduct layers
  this->blobs_.clear();
  setup_ip_layers(channels, height, width);
  for(int i = 0; i < total_class_; i ++){
      for (int j = 0; j < k_; j ++){
        add_to_learnable(ip_layer_vec_[i][j]->blobs(), this->blobs_);
      }
  }
  CHECK_EQ(this->blobs_.size(), total_class_ * k_ * 2) // each ip layer has a bias blob
    << "learnable blobs size dont match";
  // LOG(ERROR) << " blobs_ size: " << this->blobs_.size();
  // centroids
  centroids_.resize(total_class_);
  for(int i = 0; i < total_class_; i++){
       centroids_[i].resize(k_);
       for(int j = 0; j < k_; j ++){
           centroids_[i][j].reset(new Blob<Dtype>());
           centroids_[i][j]->Reshape(1, channels, height, width);
       }
       add_to_learnable(centroids_[i], this->blobs_);
  } 
  //LOG(ERROR) << " blobs_ size: " << this->blobs_.size();
  CHECK_EQ(this->blobs_.size(), total_class_ * k_ * 3)
    << "learnable blobs size dont match";

  // data
  cache_data_size_.resize(total_class_);
  cache_data_.resize(total_class_);
  cache_label_.resize(total_class_);
  for (int i = 0; i < total_class_; ++i){
    cache_data_size_[i] = 0;
    cache_data_[i].reset(new Blob<Dtype>());
    cache_label_[i].reset(new Blob<Dtype>());
    cache_data_[i]->Reshape(CLUSTERING_CACHE_DATA_SIZE_MAX_, channels, height, width);
    cache_label_[i]->Reshape(CLUSTERING_CACHE_DATA_SIZE_MAX_, 1, 1, 1);
  }
}

template <typename Dtype>
void ClusteringLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int width  = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int num  = bottom[0]->num();

  // LOG(ERROR) << " check point, blobs_ size: " << this->blobs_.size();
  // TRAIN PHASE
  if (param_seted_){
    vector<int> shape(2, 0);
    shape[0] = num;
    shape[1] = num_output_;
    top[0]->Reshape(shape);
  }
  // TEST PHASE, load weights from files,
  if (! param_seted_ ){
    param_seted_ = true;
    ClusteringParameter clustering_param = this->layer_param_.clustering_param();
    // num_output_ = this->blobs_[0]->shape()[0]; // W's shape is num_output * num_in
    num_output_ = clustering_param.num_output(); 
    across_class_ = clustering_param.across_class();
    total_class_ = across_class_ ? 1 : clustering_param.total_class();
    k_ = clustering_param.k();
    CHECK_EQ(k_ * total_class_ * 3, this->blobs_.size())
      << "clustering k dont match";
    CHECK_EQ(this->blobs_[0]->shape()[0], num_output_) 
      << "clustering layer dimension dont match";

    LOG(ERROR) << " check point, num_output: " << num_output_ 
      << " total_class_: " << total_class_ << " k " << k_;
    vector<int> shape(2, 0);
    shape[0] = num;
    shape[1] = num_output_;
    top[0]->Reshape(shape);
    setup_ip_layers(channels, height, width);
    int cnt = 0;
    for (int i = 0; i < total_class_; ++i){
      for (int j = 0; j < k_; ++j){
        ip_layer_vec_[i][j]->blobs()[0] = 
           this->blobs_[cnt++];
        ip_layer_vec_[i][j]->blobs()[1] = 
           this->blobs_[cnt++];
      }
    }
    LOG(ERROR) << " check point, cnt: " << cnt;
    LOG(ERROR) << " check point, blobs_ size: " << this->blobs_.size();
    centroids_.resize(total_class_);
    for (int i = 0; i < total_class_; ++i){
      // centroids_.resize(k_);
      for (int j = 0; j < k_; ++j){
        LOG(ERROR) << " check point, cnt: " << cnt;
        centroids_[i].push_back(this->blobs_[cnt++]);
      }
    }
    LOG(ERROR) << " check point, cnt: " << cnt;
  }
}

template <typename Dtype>
void ClusteringLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int width  = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int num  = bottom[0]->num();
  const int sz = channels * height * width;

  //LOG(ERROR) << " check point ";
  const Dtype * data  = bottom[0]->cpu_data();
  dist_.resize(num);
  assigned_centers_.resize(num); 
  // testing
  if(bottom.size() == 1){

    // 1. find the neareat
    for (int n = 0; n < num; ++n){
      Dtype dist, min_dist = 0xFFFFFFFE;
      for (int l = 0; l < total_class_; ++l){
        int idx = nearest(data + n * sz, sz, k_, centroids_[l], dist);
        if (min_dist > dist){
          min_dist = dist;
          assigned_centers_[n].first = l;
          assigned_centers_[n].second = idx;
        }
      }
      dist_[n] = min_dist;
    }
  }

  // training
  else if (bottom.size() == 2){
    // LOG(ERROR) << " check point ";
    const Dtype * label = bottom[1]->cpu_data();
    int min_size = cache_data_size_[0]; 
    for (int i = 0; i < cache_data_size_.size(); ++i){
      min_size = (min_size > cache_data_size_[i])? cache_data_size_[i] : min_size;
    }

    
    if (min_size < CLUSTERING_CACHE_DATA_SIZE_MAX_){

      // caching data
      // LOG(ERROR) << " check point caching...";
      // LOG(ERROR) << "Caching: " << vector_to_string(cache_data_size_);
      for (int i = 0; i < num; ++i){
        const int l = across_class_ ? 0 : label[i];
        int & num1 = cache_data_size_[l];
        if(num1 < CLUSTERING_CACHE_DATA_SIZE_MAX_){
          Dtype * dest = cache_data_[l]->mutable_cpu_data();
          caffe_copy<Dtype>(sz, data + i * sz, dest + num1 * sz);
          num1 ++;
        }
      }

      // update min size
      min_size = cache_data_size_[0]; 
      for (int i = 0; i < cache_data_size_.size(); ++i){
        min_size = (min_size > cache_data_size_[i])? cache_data_size_[i] : min_size;
      }

      // k means
      // LOG(ERROR) << " check point ";
      if (min_size >= CLUSTERING_CACHE_DATA_SIZE_MAX_){
        // LOG(ERROR) << " check point kmeans..." << min_size;
        // getchar();
        for(int l = 0; l < total_class_; l++){
          kmeans(cache_data_[l]->cpu_data(), cache_data_[l]->num(), sz, k_, 
            centroids_[l], cache_label_[l]->mutable_cpu_data());
        }
        LOG(ERROR) << "spliting into " << total_class_ << " x " << k_ <<" nodes";
        // for (int l = 0; l < total_class_; ++l) {
        //   for (int k = 0; k < k_; ++k) {
        //     LOG(ERROR) << blob_to_string(centroids_[l][k]->cpu_data(), sz);
        //   }
        // }
        // getchar();
      }

      // assign cluster randomly
      for (int n = 0; n < num; ++n){
        assigned_centers_[n].first = across_class_ ? 0 : label[n];
        assigned_centers_[n].second = int(caffe_rng_rand() % k_);
        dist_[n] = 0;
      }
    }
    else{ // min_size == CLUSTERING_CACHE_DATA_SIZE_MAX_
      for (int n = 0; n < num; ++n){

        // 1. find the neareat 
        const int l = across_class_ ? 0 : label[n];
        Dtype dist;
        int idx = nearest(data + n * sz, sz, k_, centroids_[l], dist);
        assigned_centers_[n].first = l;
        assigned_centers_[n].second = idx;
        dist_[n] = dist;
        // LOG(ERROR) << n <<" label " << l << " k: " << idx << "(" << k_ << ")" << " dist " << dist;
      }
    }

    // forcing no cluster, only for google testing/debug
    if (! branch_){
      for (int n = 0; n < num; ++n){
        assigned_centers_[n].first = across_class_ ? 0 : label[n];
        assigned_centers_[n].second = 0;
        dist_[n] = 0;
      }    
    }

    // // display messages
    // ostringstream stream;
    // for (int n = 0; n < num; ++n){
    //   stream << assigned_centers_[n].first 
    //     << " " << assigned_centers_[n].second << " ";
    // }
    // LOG(ERROR) << "FP " << stream.str();
  }

  // FP the corresponding branch(ip layers)
  CHECK_EQ(ip_layer_vec_[0][0]->blobs()[0]->shape()[0], num_output_) << "IP layer is wrong";
  CHECK_EQ(ip_layer_vec_[0][0]->blobs()[0]->shape()[1], sz) << "IP layer is wrong";
  // LOG(ERROR) << " check point, IP layers: " << ip_layer_vec_.size() << " x " << ip_layer_vec_[0].size();
  // LOG(ERROR) << " IP layers wieghts:" << ip_layer_vec_[0][0]->blobs()[0]->shape_string();
  // LOG(ERROR) << " IP layers bias:" << ip_layer_vec_[0][0]->blobs()[1]->shape_string();
  shared_ptr<Blob<Dtype> > ip_data_bottom, ip_data_top;
  ip_data_bottom.reset(new Blob<Dtype>(1, channels, height, width));
  ip_data_top.reset(new Blob<Dtype>());
  for (int n = 0; n < num; ++n){
    caffe_copy<Dtype>(sz, data + n * sz, ip_data_bottom->mutable_cpu_data());
    ip_bottom_vec_.resize(1);
    ip_top_vec_.resize(1);
    ip_bottom_vec_[0] = ip_data_bottom.get();
    ip_top_vec_[0] = ip_data_top.get();
    ip_layer_vec_[assigned_centers_[n].first][assigned_centers_[n].second]->Reshape(ip_bottom_vec_, ip_top_vec_);
    CHECK_EQ(ip_top_vec_[0]->count(), num_output_) << "ip layer channel is wrong";
    ip_layer_vec_[assigned_centers_[n].first][assigned_centers_[n].second]->Forward(ip_bottom_vec_, ip_top_vec_);
    caffe_copy<Dtype>(num_output_, ip_data_top->cpu_data(), top[0]->mutable_cpu_data() + n * num_output_);
  }
}

template <typename Dtype>
void ClusteringLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const int width  = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int num  = bottom[0]->num();
  const int sz = channels * height * width;
  const Dtype * bottom_data = bottom[0]->cpu_data();
       
        Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype * top_diff = top[0]->cpu_diff();

  // BP through ip layers
  CHECK_EQ(assigned_centers_.size(), num)
    << "data size dont match";
  CHECK_EQ(top[0]->channels(), num_output_)
    << "data size dont match";

  // BP the corresponding branch
  shared_ptr<Blob<Dtype> > ip_data_bottom, ip_data_top;
  ip_data_bottom.reset(new Blob<Dtype>(1, channels, height, width));
  ip_data_top.reset(new Blob<Dtype>(1, num_output_, 1, 1));
  ip_bottom_vec_.resize(1);
  ip_top_vec_.resize(1);
  ip_bottom_vec_[0] = ip_data_bottom.get();
  ip_top_vec_[0] = ip_data_top.get();
  vector<bool> pd;
  pd.push_back(true);
  for (int n = 0; n < num; ++n){
    const int l = assigned_centers_[n].first;
    const int k = assigned_centers_[n].second;
    caffe_set<Dtype>(num_output_, Dtype(0.0), ip_top_vec_[0]->mutable_cpu_diff());
    caffe_copy<Dtype>(num_output_, top_diff + n * num_output_, ip_top_vec_[0]->mutable_cpu_diff());
    caffe_copy<Dtype>(sz, bottom_data+n*sz, ip_bottom_vec_[0]->mutable_cpu_data());
    caffe_set<Dtype>(sz, Dtype(0.0), ip_bottom_vec_[0]->mutable_cpu_diff());
    ip_layer_vec_[l][k]->Backward(ip_top_vec_, pd, ip_bottom_vec_);
    caffe_copy<Dtype>(sz, ip_bottom_vec_[0]->cpu_diff(), bottom_diff + n * sz);
  }

  // // display messages
  // ostringstream stream;
  // for (int n = 0; n < num; ++n){
  //   stream << assigned_centers_[n].first 
  //     << " " << assigned_centers_[n].second << " ";
  // }
  // LOG(ERROR) << "BP " << stream.str();

  // BP through kmeans, this is optianal 
  shared_ptr<Blob<Dtype> > tmp;
  tmp.reset(new Blob<Dtype>(1, channels, height, width));
  for (int n = 0; n < num; ++n){
    const int l = assigned_centers_[n].first;
    const int k = assigned_centers_[n].second;
    caffe_sub<Dtype>(sz, bottom_data+n*sz, centroids_[l][k]->cpu_data(), tmp->mutable_cpu_data());
    caffe_cpu_axpby<Dtype>(sz, Dtype(lambda_ * dist_[n]), tmp->cpu_data(), Dtype(1.0), bottom_diff + n * sz);
  }

}

template <typename Dtype>
void ClusteringLayer<Dtype>::setup_ip_layers(int channels, int height, int width){
  LayerParameter inner_param;
  inner_param.mutable_inner_product_param()->set_num_output(num_output_);
  inner_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
  inner_param.mutable_inner_product_param()->set_bias_term(true);
  ip_layer_vec_.resize(total_class_);
  shared_ptr<Blob<Dtype> > blob_bottom, blob_top;
  blob_bottom.reset(new Blob<Dtype>()); blob_top.reset(new Blob<Dtype>());
  blob_bottom->Reshape(1, channels, height, width);
  ip_top_vec_.resize(1);
  ip_bottom_vec_.resize(1);
  ip_top_vec_[0] = blob_top.get();
  ip_bottom_vec_[0] = blob_bottom.get();
  for(int i = 0; i < total_class_; i ++){
      ip_layer_vec_[i].resize(k_);
      for (int j = 0; j < k_; j ++){
        ip_layer_vec_[i][j].reset(new InnerProductLayer<Dtype>(inner_param));
        ip_layer_vec_[i][j]->SetUp(ip_bottom_vec_, ip_top_vec_);
      }
  }
}

template <typename Dtype>
void ClusteringLayer<Dtype>::kmeans(const Dtype * data, int n, int m, int k, 
  vector<shared_ptr<Blob<Dtype> > > & centroids, Dtype * labels){
  CHECK_EQ(centroids.size(), k) << "centroids size";
  CHECK_EQ(centroids[0]->count(), m) << "dimension";
  vector<int> counts; counts.resize(k);
  double old_error, error = 0xFFFFFFFE;
  const double t = 1E-8;
  const int MAX_ITERS = 100;

  // initialization
  //LOG(ERROR) << "check point init";
  vector<shared_ptr<Blob<Dtype> > > ctmp;
  ctmp.resize(k);
  vector<int> shape = centroids[0]->shape();
  for (int i = 0; i < k; ++i){
    ctmp[i].reset(new Blob<Dtype>(shape));
    // int h = n / k * i;
    // caffe_copy<Dtype>(m, data + h * m, centroids[i]->mutable_cpu_data());
  }
  kmpp(data, n, m, k, centroids);

  shared_ptr<Blob<Dtype> > tmp;
  tmp.reset(new Blob<Dtype>(shape));

  //LOG(ERROR) << "check point main loop";
  // main loop
  int iters = 0;
  do {
      old_error = error, error = 0;

      // clear old counts and temp centroids
      for (int i = 0; i < k; counts[i++] = 0) {
         caffe_set<Dtype>(m, Dtype(0.0), ctmp[i]->mutable_cpu_data());
      }

      //LOG(ERROR) << "check point assign";
      for (int h = 0; h < n; h++) {
         // cur
         const Dtype * cur = data + h * m;
         double min_distance = 0xFFFFFFFE;
         for (int i = 0; i < k; i++) {
            double distance = cal_dist(m, cur, centroids[i]->cpu_data(), tmp.get());
            // double distance2 = cal_dist(m, cur, centroids[i]->cpu_data(), tmp->mutable_cpu_data());
            // LOG(ERROR) << blob_to_string(cur, m);
            // LOG(ERROR) << blob_to_string(centroids[i]->cpu_data(), m);
            // LOG(ERROR) << "Center" << i << ":" << distance << " 2 " << distance2;
            // getchar();
            if (distance < min_distance) {
               labels[h] = i;
               min_distance = distance;
            }
         }
         // update
         caffe_add<Dtype>(m, cur, ctmp[labels[h]]->cpu_data(), ctmp[labels[h]]->mutable_cpu_data());
         counts[labels[h]]++;
         error += min_distance;
      }

      //LOG(ERROR) << "check point centroids";
      for (int i = 0; i < k; i++) { /* update all centroids */
        Dtype alpha = counts[i] ? 1.0 / counts[i] : 1.0;
        caffe_cpu_axpby<Dtype>(m, alpha, ctmp[i]->cpu_data(), Dtype(0.0), centroids[i]->mutable_cpu_data());
        // LOG(ERROR) << "Number: " << counts[i];
      }
      // LOG(ERROR) << "K means: " << iters << " " << fabs(error - old_error) 
      //   << " Error " << error << " nums: " << vector_to_string(counts);
      // getchar();
  } while (fabs(error - old_error) > t && ++iters < MAX_ITERS);

}

template <typename Dtype>
void ClusteringLayer<Dtype>::kmpp(const Dtype * data, int n, int m, int k, 
  vector<shared_ptr<Blob<Dtype> > > & centroids){
// kmeans ++ initialization
  int i = caffe_rng_rand() % n;
  caffe_copy<Dtype>(m, data + i * m, centroids[0]->mutable_cpu_data());
  shared_ptr<Blob<Dtype> > tmp;
  tmp.reset(new Blob<Dtype>(1, m, 1, 1));
  for (int i = 1; i < k; ++i){
    double max_min_dist = 0;
    int max_farest = 0;
    for (int j = 0; j < n; ++j){
      double min_dist = 0xFFFFFFFE;
      for (int r = 0; r < i; ++r){
        double dist = cal_dist(m, data + j * m, centroids[r]->cpu_data(), tmp->mutable_cpu_data());
        if (min_dist > dist) {
          min_dist = dist;
        }
      }
      if (max_min_dist < min_dist){
        max_min_dist = min_dist;
        max_farest = j;
      }
    }
    caffe_copy<Dtype>(m, data + max_farest * m, centroids[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
int ClusteringLayer<Dtype>::nearest(const Dtype * data, int m, int k, 
    vector<shared_ptr<Blob<Dtype> > > & centroids, Dtype & dist){

  CHECK_EQ(k, centroids.size()) << "clusters number is wrong";
  double disttmp = 0;
  int idx = 0;
  dist = 0xFFFFFFFE;
  vector<int> shape = centroids[0]->shape();
  shared_ptr<Blob<Dtype> > tmp;
  tmp.reset(new Blob<Dtype>(shape));
  for (int i = 0; i < k; ++i){
    disttmp = cal_dist(m, data, centroids[i]->cpu_data(), tmp->mutable_cpu_data());
    if (disttmp < dist){
      dist = disttmp;
      idx = i;
    }
  }
  return idx;
}


INSTANTIATE_CLASS(ClusteringLayer);
REGISTER_LAYER_CLASS(Clustering);

}  // namespace caffe
