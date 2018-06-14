#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_add_layer.hpp"

/*
这个层的操作还是比较简单的，就是在归一化后的weight*feature之后，
对应的减去margin的操作，也就是amsoftmax论文的中的x-m操作
求导也很简单，对x求导梯度为1
*/
namespace caffe {
  // layer setup 读取层参数，一般只提供bias_即可
  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificAddParameter& param = this->layer_param_.label_specific_add_param();
    bias_ = param.bias();
    transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
    anneal_bias_ = param.has_bias_base();
    bias_base_ = param.bias_base();
    bias_gamma_ = param.bias_gamma();
    bias_power_ = param.bias_power();
    bias_min_ = param.bias_min();
    bias_max_ = param.bias_max();
    iteration_ = param.iteration();
  }

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    if(top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
    if (top.size() == 2) top[1]->Reshape({ 1 }); // extended initializer lists only available with -std=c++11
  }

template <typename Dtype>
void LabelSpecificAddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  //两个输入bottom， data from previous fc layer and label data 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  // dim是输入数据除去num的维度，对应于这里其实就是fc layer的输出个数，即类别数。因为上一层全连接层每一个样本输出一个对应于全部类别数的向量，后面转化为概率
  int dim = count / num; 
  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

  if (!transform_test_ && this->phase_ == TEST) return;
  // 是否通过模拟退火算来动态改变bias_，一般设置为false，不用管
  if (anneal_bias_) {
    bias_ = bias_base_ + pow(((Dtype)1. + bias_gamma_ * iteration_), bias_power_) - (Dtype)1.;
    bias_ = std::max(bias_, bias_min_);
    bias_ = std::min(bias_, bias_max_);
    iteration_++;
  }
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = bias_;
  }
  //核心代码就这三行，将当前类的ground truth对应位置的值减去bias_
  // top_data里面保存的已经是归一化之后的weight和feature的乘积，即cos(theta)
  // top-data中每一个样本都是一个n维度的值，对应着n个类别的值，找到当前这个样本对应的真实类别位置，减去bias即可
  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    if(top_data[i * dim + gt] > -bias_) top_data[i * dim + gt] += bias_;
  }
}

template <typename Dtype>
void LabelSpecificAddLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  if (top[0] != bottom[0] && propagate_down[0]) {
    //反向传播值不变，梯度不变，因为执行的是x-bias的操作，对x求导是1，直接bottom_diff = top_diff即可
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int count = bottom[0]->count();
    caffe_copy(count, top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificAddLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificAddLayer);
REGISTER_LAYER_CLASS(LabelSpecificAdd);

}  // namespace caffe
