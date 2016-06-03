#include <vector>
#include <utility>
#include "boost/make_shared.hpp"
using boost::make_shared;
#include "caffe/filler.hpp"
#include "caffe/layers/clustering_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <fstream>

namespace caffe {

template <typename Dtype>
void ClusteringLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ClusteringParameter clustering_param = this->layer_param_.clustering_param();
  k_            = clustering_param.k();
  total_class_  = clustering_param.total_class();
  lambda_       = clustering_param.lambda();  //current layers' loss weight
  branch_       = clustering_param.branch();
  across_class_ = clustering_param.across_class();
  data_size_    = clustering_param.data_size();
  total_class_  = across_class_ ? 1 : clustering_param.total_class();
  dominate_     = clustering_param.dominate();
  total_class_  = dominate_ >= 0 ? 1 : total_class_;
  num_layer_    = clustering_param.num_layer();
  param_seted_  = true;
  soft_         = clustering_param.soft();
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
  // setup_ip_layers(channels, height, width);
  setup_mip_layers(channels, height, width);
  for(int i = 0; i < total_class_; i ++){
      for (int j = 0; j < k_; j ++){
        add_to_learnable(mip_layer_vec_[i][j]->blobs(), this->blobs_);
      }
  }
  // each mip has num_layer_ * 2 blobs
  // totaly there are total_class_ * k_ branches
  const int num_blobs_mip = total_class_ * k_ * 2 * num_layer_;
  CHECK_EQ(this->blobs_.size(), num_blobs_mip) // each mip layer has a bias blob
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
  CHECK_EQ(this->blobs_.size(), num_blobs_mip + total_class_ * k_)
    << "learnable blobs size dont match";

  // data
  cache_data_size_.resize(total_class_);
  cache_data_.resize(total_class_);
  cache_label_.resize(total_class_);
  for (int i = 0; i < total_class_; ++i){
    cache_data_size_[i] = 0;
    cache_data_[i].reset(new Blob<Dtype>());
    cache_label_[i].reset(new Blob<Dtype>());
    cache_data_[i]->Reshape(data_size_, channels, height, width);
    cache_label_[i]->Reshape(data_size_, 1, 1, 1);
  }
}

template <typename Dtype>
void ClusteringLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int width    = bottom[0]->width();
  const int height   = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int num      = bottom[0]->num();

  // LOG(ERROR) << " check point, blobs_ size: " << this->blobs_.size();

  // TEST PHASE, load weights from files, and setup the inner product layer
  if (! param_seted_ ){

    param_seted_ = true;
    ClusteringParameter clustering_param = this->layer_param_.clustering_param();
    // num_output_ = this->blobs_[0]->shape()[0]; // W's shape is num_output * num_in
    num_output_   = clustering_param.num_output(); 
    across_class_ = clustering_param.across_class();
    total_class_  = across_class_ ? 1 : clustering_param.total_class();
    k_            = clustering_param.k();
    data_size_    = clustering_param.data_size();
    dominate_     = clustering_param.dominate();
    total_class_  = dominate_ >= 0 ? 1 : total_class_;
    num_layer_    = clustering_param.num_layer();

    const int num_blobs_mip = num_layer_ * total_class_ * k_ * 2;
    const int num_blobs_tot = total_class_ * k_ + num_blobs_mip;
    CHECK_EQ(num_blobs_tot, this->blobs_.size())
      << "clustering k dont match";
    CHECK_EQ(this->blobs_[0]->shape()[0], num_output_) 
      << "clustering layer dimension dont match";
    // LOG(ERROR) << " check point, num_output: " << num_output_ 
    //   << " total_class_: " << total_class_ << " k " << k_;

    // assigning weights for all the branches
    setup_mip_layers(channels, height, width);
    int cnt = 0;
    for (int i = 0; i < total_class_; ++ i){
      for (int j = 0; j < k_; ++ j){
        for (int b = 0; b < num_layer_; ++ b){
          mip_layer_vec_[i][j]->blobs()[b * 2]     = this->blobs_[cnt++];
          mip_layer_vec_[i][j]->blobs()[b * 2 + 1] = this->blobs_[cnt++];
        }
      }
    }
    // LOG(ERROR) << " check point, cnt: " << cnt;
    // LOG(ERROR) << " check point, blobs_ size: " << this->blobs_.size();

    // initialize centroids
    centroids_.resize(total_class_);
    for (int i = 0; i < total_class_; ++i){
      // centroids_.resize(k_);
      for (int j = 0; j < k_; ++j){
        // LOG(ERROR) << " check point, cnt: " << cnt;
        centroids_[i].push_back(this->blobs_[cnt++]);
      }
    }
    CHECK_EQ(cnt, num_blobs_tot) << "clustering k dont match"; 
    centroids_init_ = true;
    // LOG(ERROR) << " check point, cnt: " << cnt;
  }
  vector<int> shape(2, 0);
  shape[0] = num;
  shape[1] = num_output_;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ClusteringLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const int num      = bottom[0]->num();
  const int width    = bottom[0]->width();
  const int height   = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int sz       = channels * height * width;
  const Dtype * data = bottom[0]->cpu_data();

  dist_.resize(num);
  assigned_centers_.resize(num); 
  // testing
  if(bottom.size() == 1 || this->phase_ == TEST){

    // 1. find the neareat
    // std::ofstream of("cluster.txt", ios::app);
    vector<vector<vector<Dtype> > > all_dists(num);
    for (int n = 0; n < num; ++n){
      all_dists[n].resize(total_class_);
      Dtype dist, min_dist = 0xFFFFFFFE;
      for (int l = 0; l < total_class_; ++l){
        all_dists[n][l].resize(k_);
        int idx = nearest(data + n * sz, sz, k_, centroids_[l], all_dists[n][l]);
        dist = all_dists[n][l][idx];
        if (min_dist > dist){
          min_dist = dist;
          assigned_centers_[n].first  = l;
          assigned_centers_[n].second = idx;
        }
      }
      dist_[n] = min_dist;
      // of << assigned_centers_[n].second << "\n";
    }
    // Forward Branches
    if (! soft_){
      shared_ptr<Blob<Dtype> > ip_data_bottom, ip_data_top;
      ip_data_bottom.reset(new Blob<Dtype>(1, channels, height, width));
      ip_data_top.reset(new Blob<Dtype>());
      vector<Blob<Dtype>*> ip_top_vec(1, ip_data_top.get());
      vector<Blob<Dtype>*> ip_bottom_vec(1, ip_data_bottom.get());
      for (int n = 0; n < num; ++n){
        caffe_copy(sz, data + n * sz, ip_bottom_vec[0]->mutable_cpu_data());
        mip_layer_vec_[assigned_centers_[n].first][assigned_centers_[n].second]->Forward(ip_bottom_vec, ip_top_vec);
        caffe_copy(num_output_, ip_top_vec[0]->cpu_data(), top[0]->mutable_cpu_data() + n * num_output_);
      }
    }else{ //soft 
      // forward all branches
      vector<shared_ptr<Blob<Dtype> > > ip_data_tops(total_class_ * k_);
      vector<Blob<Dtype>*> ip_top_vec(1, NULL);
      for (int i = 0; i < ip_data_tops.size(); ++i){
        ip_data_tops[i].reset(new Blob<Dtype>());
      }
      for (int l = 0; l < total_class_; ++l){
        for (int k = 0; k < k_; ++k){
          ip_top_vec[0] = ip_data_tops[l * k_ + k].get();
          mip_layer_vec_[l][k]->Forward(bottom, ip_top_vec);
        }
      }
      // norm p
      for (int n = 0; n < num; ++n){
        double sum = 0;
        for (int l = 0; l < total_class_; ++l)
          for (int k = 0; k < k_; ++k)
            sum += 1.0 / all_dists[n][l][k];

        for (int l = 0; l < total_class_; ++l)
          for (int k = 0; k < k_; ++k)
            all_dists[n][l][k] = (1.0 / all_dists[n][l][k]) / sum;
      }
      // weighted sum
      for (int n = 0; n < num; ++n)
        for (int l = 0; l < total_class_; ++l)
          for (int k = 0; k < k_; ++k){
            caffe_cpu_axpby(num_output_, 
              Dtype(all_dists[n][l][k]), ip_data_tops[k + l * k_]->cpu_data() + n * num_output_,
              Dtype(0), top[0]->mutable_cpu_data() + n * num_output_);
          }
    } //soft assignment end
  } // testing end

  // training
  else if (bottom.size() == 2 && this->phase_ == TRAIN){
    // LOG(ERROR) << " check point ";
    const Dtype * label = bottom[1]->cpu_data();
    int min_size = cache_data_size_[0]; 
    for (int i = 0; i < cache_data_size_.size(); ++i){
      min_size = (min_size > cache_data_size_[i])? cache_data_size_[i] : min_size;
    }

    if (min_size < data_size_){

      // caching data
      // LOG(ERROR) << " check point caching...";
      // LOG(ERROR) << "Caching: " << vector_to_string(cache_data_size_);
      for (int i = 0; i < num; ++i){
        const int l = (across_class_ || dominate_ >= 0) ? 0 : label[i];
        if (dominate_ >= 0 && label[i] != dominate_){
          continue;
        }
        int & num1 = cache_data_size_[l];
        if(num1 < data_size_){
          Dtype * dest = cache_data_[l]->mutable_cpu_data();
          caffe_copy(sz, data + i * sz, dest + num1 * sz);
          num1 ++;
        }
      }

      // update min size
      min_size = cache_data_size_[0]; 
      for (int i = 0; i < cache_data_size_.size(); ++i){
        min_size = (min_size > cache_data_size_[i])? cache_data_size_[i] : min_size;
      }

      // do k means
      // LOG(ERROR) << " check point ";
      if (min_size >= data_size_){
        // LOG(ERROR) << " check point kmeans..." << min_size;
        // getchar();
        double error = 0, sb = 0;
        vector<int> counts;
        for(int l = 0; l < total_class_; l++){
          error += kmeans(cache_data_[l]->cpu_data(), cache_data_[l]->num(), sz, k_, 
            centroids_[l], cache_label_[l]->mutable_cpu_data(), counts, centroids_init_);
          sb += Sb_cluster(centroids_[l]);

          ostringstream stream; stream << "K " << counts.size() << " (";
          for (int i = 0; i < counts.size(); ++i){
            stream << counts[i] << " ";
          }
          stream << ")";
          LOG(ERROR) << stream.str();
          // !!!
          // cache_data_size_[l] = 0;  // clear for next
        }
        centroids_init_ = true;
        LOG(ERROR) << "Layer "<< this->layer_param_.name() << " K Means split into " 
          << total_class_ << " x " << k_ <<" nodes, error: " << error  
          << " error(ref): " << error / (sb);
        // for (int l = 0; l < total_class_; ++l) {
        //   for (int k = 0; k < k_; ++k) {
        //     LOG(ERROR) << blob_to_string(centroids_[l][k]->cpu_data(), sz);
        //   }
        // }
        // getchar();
      }

      // assign cluster
      if (centroids_init_){
        for (int n = 0; n < num; ++n){

          // 1. find the neareat 
          const int l = (across_class_ || dominate_ >= 0) ? 0 : label[n];
          vector<Dtype> dists;
          int idx = nearest(data + n * sz, sz, k_, centroids_[l], dists);
          assigned_centers_[n].first  = l;
          // assigned_centers_[n].second = idx;
          assigned_centers_[n].second = mc_infer(dists);
          // LOG(ERROR) << "idx " << assigned_centers_[n].second;
          CHECK_LE(assigned_centers_[n].second, k_) << "mc error";
          dist_[n] = dists[idx];
          // LOG(ERROR) << n <<" label " << l << " k: " << idx << "(" << k_ << ")" << " dist " << dist;
        }
      }else{
        for (int n = 0; n < num; ++n){
          assigned_centers_[n].first = (across_class_ || dominate_ >= 0) ? 0 : label[n];
          assigned_centers_[n].second = int(caffe_rng_rand() % k_);
          dist_[n] = 0;
        }
      }
    }
    else{ // min_size == data_size_
      for (int n = 0; n < num; ++n){

        // 1. find the neareat 
        const int l = (across_class_ || dominate_ >= 0) ? 0 : label[n];
        vector<Dtype> dists;
        int idx = nearest(data + n * sz, sz, k_, centroids_[l], dists);
        assigned_centers_[n].first = l;
        // assigned_centers_[n].second = idx;
        assigned_centers_[n].second = mc_infer(dists);
        CHECK_LE(assigned_centers_[n].second, k_) << "mc error";
        dist_[n] = dists[idx];
        // LOG(ERROR) << n <<" label " << l << " k: " << idx << "(" << k_ << ")" << " dist " << dist;
      }
    }

    // forcing no cluster, it's only used for google testing/debug
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

    // FP the corresponding branch(ip layers)
    // LOG(ERROR) << " check point, IP layers: " << mip_layer_vec_.size() << " x " << mip_layer_vec_[0].size();
    // LOG(ERROR) << " IP layers wieghts:" << mip_layer_vec_[0][0]->blobs()[0]->shape_string();
    // LOG(ERROR) << " IP layers bias:" << mip_layer_vec_[0][0]->blobs()[1]->shape_string();
    shared_ptr<Blob<Dtype> > ip_data_bottom, ip_data_top;
    ip_data_bottom.reset(new Blob<Dtype>(1, channels, height, width));
    ip_data_top.reset(new Blob<Dtype>());
    ip_bottom_vec_.resize(1);
    ip_top_vec_.resize(1);
    for (int n = 0; n < num; ++n){
      caffe_copy(sz, data + n * sz, ip_data_bottom->mutable_cpu_data());
      ip_bottom_vec_[0] = ip_data_bottom.get();
      ip_top_vec_[0] = ip_data_top.get();
      mip_layer_vec_[assigned_centers_[n].first][assigned_centers_[n].second]->Forward(ip_bottom_vec_, ip_top_vec_);
      caffe_copy(num_output_, ip_data_top->cpu_data(), top[0]->mutable_cpu_data() + n * num_output_);
    }
  } // training end
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
    caffe_set(num_output_, Dtype(0.0), ip_top_vec_[0]->mutable_cpu_diff());
    caffe_copy(num_output_, top_diff + n * num_output_, ip_top_vec_[0]->mutable_cpu_diff());
    caffe_copy(sz, bottom_data+n*sz, ip_bottom_vec_[0]->mutable_cpu_data());
    caffe_set(sz, Dtype(0.0), ip_bottom_vec_[0]->mutable_cpu_diff());
    mip_layer_vec_[l][k]->Backward(ip_top_vec_, pd, ip_bottom_vec_);
    caffe_copy(sz, ip_bottom_vec_[0]->cpu_diff(), bottom_diff + n * sz);
  }

  // // display messages
  // ostringstream stream;
  // for (int n = 0; n < num; ++n){
  //   stream << assigned_centers_[n].first 
  //     << " " << assigned_centers_[n].second << " ";
  // }
  // LOG(ERROR) << "BP " << stream.str();

  // BP through kmeans, this is optianal 
  if (! centroids_init_)
      return;
  Dtype grad_max = bottom[0]->asum_diff() / bottom[0]->count();
  shared_ptr<Blob<Dtype> > tmp;
  tmp.reset(new Blob<Dtype>(1, channels, height, width));
  for (int n = 0; n < num; ++n){
    const int l = assigned_centers_[n].first;
    const int k = assigned_centers_[n].second;
    caffe_sub(sz, bottom_data+n*sz, centroids_[l][k]->cpu_data(), tmp->mutable_cpu_data());
    caffe_sign(sz, tmp->cpu_data(), tmp->mutable_cpu_data(), Dtype(1E-3));
    Dtype gradient = lambda_ * dist_[n];
    gradient = grad_max < gradient ? grad_max : gradient;
    caffe_cpu_axpby(sz, gradient, tmp->cpu_data(), Dtype(1.0), bottom_diff + n * sz);
    // for (int i = 0; i < total_class_; ++i){
    //   for (int j = 0; j < k_; ++j) {
    //     if (l == i && j == k){
    //       continue;
    //     }
    //     caffe_sub<Dtype>(sz, centroids_[i][j]->cpu_data(), bottom_data+n*sz, tmp->mutable_cpu_data());
    //     caffe_sign<Dtype> (sz, tmp->cpu_data(), tmp->mutable_cpu_data(), Dtype(1E-1));
    //     Dtype gradient = lambda_ * dist_[n] * 0.00001;
    //     gradient = grad_max < gradient ? grad_max : gradient;
    //     caffe_cpu_axpby<Dtype>(sz, gradient, tmp->cpu_data(), Dtype(1.0), bottom_diff + n * sz);
    //   }
    // }
  }

}

template <typename Dtype>
void ClusteringLayer<Dtype>::setup_mip_layers(int channels, int height, int width){

  LayerParameter layer_param;
  MultipleInnerProductParameter* mip_param = layer_param.mutable_multiple_inner_product_param();
  mip_param->set_num_layer(num_layer_);
  for (int i = 0; i < num_layer_; ++i){
    mip_param->add_num_outputs(num_output_);
  }

  shared_ptr<Blob<Dtype> > blob_bottom, blob_top;
  blob_bottom.reset(new Blob<Dtype>()); blob_top.reset(new Blob<Dtype>());
  blob_bottom->Reshape(1, channels, height, width);
  vector<Blob<Dtype>*> ip_bottom_vec(1, NULL), ip_top_vec(1, NULL);
  ip_top_vec[0]    = blob_top.get();
  ip_bottom_vec[0] = blob_bottom.get();

  mip_layer_vec_.resize(total_class_);
  for(int i = 0; i < total_class_; i ++){
      mip_layer_vec_[i].resize(k_);
      for (int j = 0; j < k_; j ++){
        mip_layer_vec_[i][j].reset(new MultipleInnerProductLayer<Dtype>(layer_param));
        mip_layer_vec_[i][j]->SetUp(ip_bottom_vec, ip_top_vec);
      }
  }
}

template <typename Dtype>
void ClusteringLayer<Dtype>::setup_ip_layers(int channels, int height, int width){

  LayerParameter shared_ip_param;
  shared_ip_param.mutable_inner_product_param()->set_num_output(num_output_);
  shared_ip_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
  shared_ip_param.mutable_inner_product_param()->set_bias_term(true);

  shared_ptr<Blob<Dtype> > blob_bottom, blob_top;
  blob_bottom.reset(new Blob<Dtype>()); blob_top.reset(new Blob<Dtype>());
  blob_bottom->Reshape(1, channels, height, width);
  ip_top_vec_.resize(1);
  ip_bottom_vec_.resize(1);
  ip_top_vec_[0]    = blob_top.get();
  ip_bottom_vec_[0] = blob_bottom.get();

  ip_layer_vec_.resize(total_class_);
  for(int i = 0; i < total_class_; i ++){
      ip_layer_vec_[i].resize(k_);
      for (int j = 0; j < k_; j ++){
        ip_layer_vec_[i][j].reset(new InnerProductLayer<Dtype>(shared_ip_param));
        ip_layer_vec_[i][j]->SetUp(ip_bottom_vec_, ip_top_vec_);
      }
  }

}

template <typename Dtype>
double ClusteringLayer<Dtype>::kmeans(const Dtype * data, int n, int m, int k, 
  vector<shared_ptr<Blob<Dtype> > > & centroids, Dtype * labels,
  vector<int> & counts, bool continue_cluster){
  CHECK_EQ(centroids.size(), k) << "centroids size";
  CHECK_EQ(centroids[0]->count(), m) << "dimension";
  
  counts.resize(k);
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
  if (! continue_cluster){
    kmpp(data, n, m, k, centroids);
  }
  

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
         caffe_add(m, cur, ctmp[labels[h]]->cpu_data(), ctmp[labels[h]]->mutable_cpu_data());
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
  return error / n;
}

template <typename Dtype>
void ClusteringLayer<Dtype>::kmpp(const Dtype * data, int n, int m, int k, 
  vector<shared_ptr<Blob<Dtype> > > & centroids){
// kmeans ++ initialization
  int i = caffe_rng_rand() % n;
  caffe_copy(m, data + i * m, centroids[0]->mutable_cpu_data());
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
    caffe_copy(m, data + max_farest * m, centroids[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
int ClusteringLayer<Dtype>::nearest(const Dtype * data, int m, int k, 
    vector<shared_ptr<Blob<Dtype> > > & centroids, vector<Dtype> & dists){

  CHECK_EQ(k, centroids.size()) << "clusters number is wrong";
  double disttmp = 0;
  int idx = 0;
  dists.resize(k);
  Dtype dist = 0xFFFFFFFE;
  vector<int> shape = centroids[0]->shape();
  shared_ptr<Blob<Dtype> > tmp;
  tmp.reset(new Blob<Dtype>(shape));
  for (int i = 0; i < k; ++i){
    disttmp = cal_dist(m, data, centroids[i]->cpu_data(), tmp->mutable_cpu_data());
    dists[i] = disttmp;
    if (disttmp < dist){
      dist = disttmp;
      idx = i;
    }
  }
  return idx;
}

template <typename Dtype>
int ClusteringLayer<Dtype>::mc_infer(vector<Dtype> & dists)
{
  vector<Dtype> p, cdf;
  p.resize(dists.size());
  cdf.resize(dists.size());
  double sum = 0;
  for (int i = 0; i < dists.size(); ++i){
    // p[i] = exp(- dists[i] );
    p[i] = 1 / (dists[i] + 0.001);
    sum += p[i];
  }
  for (int i = 0; i < dists.size(); ++i){
    p[i] /= sum;
  }
  cdf[0] = p[0];
  for (int i = 1; i < dists.size(); ++i){
    cdf[i] = cdf[i-1] + p[i];
  }
  double r = (caffe_rng_rand() + 0.0) / UINT_MAX;
  for (int i = 0; i < dists.size(); ++i){
    if (r < cdf[i]){
      return i;
    }
  }
  return 0;
}

INSTANTIATE_CLASS(ClusteringLayer);
REGISTER_LAYER_CLASS(Clustering);

}  // namespace caffe
