// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <error.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <sys/mman.h>
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/calcu_pthread.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

typedef struct {
    float* bottom_data;
    float* bottom_rois;
    float* top_data;
    int    num_rois;
    int    top_count;
    int    pooled_height_;
    int    pooled_width_;
    int    width_;
    int    height_;
    int    channels_;
    int*   argmax_data;
    float  spatial_scale_;
} roi_pool_arg_t;

pthread_mutex_t counter_lock;
static int counter = -1;
static __inline unsigned int get_counter(void)
{
    pthread_mutex_lock(&counter_lock);
    counter++;
    pthread_mutex_unlock(&counter_lock);
    return (unsigned int)counter;
}

void roi_pool_inner_thread(void* set_args, int pos)
{
  roi_pool_arg_t* args     = (roi_pool_arg_t*)set_args;
  const float* bottom_data    = args -> bottom_data;
  const float* bottom_rois    = args -> bottom_rois;
  float*       top_data       = args -> top_data;
  int          num_rois       = args -> num_rois;
  int          top_count      = args -> top_count;
  int          pooled_height_ = args -> pooled_height_;
  int          pooled_width_  = args -> pooled_width_;
  int          width_         = args -> width_;
  int          height_        = args -> height_;
  int          channels_      = args -> channels_;
  int*         argmax_data    = args -> argmax_data;
  float        spatial_scale_ = args -> spatial_scale_;

  int in_area  = height_        * width_;
  int out_area = pooled_height_ * pooled_width_;
  
  while (1)
  {
    int n = get_counter();

    if(n >= num_rois) break;
    
    //int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[n*5 + 1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[n*5 + 2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[n*5 + 3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[n*5 + 4] * spatial_scale_);
    //printf("n:%d roi_batch_ind:%d bottom_rois[0]:%f bottom_rois:%d\r\n",n,roi_batch_ind,bottom_rois[0],bottom_rois);
    //printf("roi_start_w:%d roi_start_h:%d roi_end_w:%d roi_end_h:%d\r\n",roi_start_w,roi_start_h,roi_end_w,roi_end_h);
    //CHECK_GE(roi_batch_ind, 0);
    //CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
    const float bin_size_h = (float)(roi_height) / (float)(pooled_height_);
    const float bin_size_w = (float)(roi_width) / (float)(pooled_width_);

    //const float* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    const float* batch_data = bottom_data;
    
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = (int)(floor((float)(ph) * bin_size_h));
          int wstart = (int)(floor((float)(pw) * bin_size_w));
          int hend = (int)(ceil((float)(ph + 1) * bin_size_h));
          int wend = (int)(ceil((float)(pw + 1) * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pre_pool_index = n * channels_ * out_area + c * out_area;
          const int pool_index     = pre_pool_index + ph * pooled_width_ + pw;
          //printf("c:%d n:%d pool_index:%d batch_offset:%d\r\n",c,n,(pool_index),c * in_area);
          if (is_empty) 
          {
            top_data[pool_index]    = 0;
            argmax_data[pool_index] = -1;
          }

          register int   max_index = 0;
          register float out_data  = 0;
          for (int h = hstart; h < hend; ++h) 
          {
            for (int w = wstart; w < wend; ++w) 
            {
              const int pre_index = c * in_area;
              const int index     = pre_index + h * width_ + w;
              if (batch_data[index] > out_data)
              {
                out_data    = batch_data[index];
                max_index   = index;
              }
            }
          }
          top_data[pool_index]    = out_data;
          argmax_data[pool_index] = max_index;
        }
      }
    }
  }
  
}

namespace caffe {

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

double roipool_elapsed_time=0;
template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 struct timespec start, finish;
clock_gettime(CLOCK_MONOTONIC, &start);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

//  printf("#channels_:%d##height_:%d  width_:%d bottom[1]->offset(1):%d bottom[0]->offset(0, 1):%d top[0]->offset(0, 1):%d bottom[0]->offset(0):%d\r\n",channels_,height_,width_,bottom[1]->offset(1),bottom[0]->offset(0, 1),top[0]->offset(0, 1),bottom[0]->offset(0));

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  
    queue_t           queue_Q[MAX_CPU_NUMBER];
    roi_pool_arg_t    ins_args;
    
    ins_args.bottom_data    =  (float*)(bottom[0] -> cpu_data());
    ins_args.bottom_rois    =  (float*)(bottom[1] -> cpu_data());
    ins_args.top_data       =  (float*)(top[0] -> mutable_cpu_data());
    ins_args.num_rois       =  bottom[1] -> num();
    ins_args.top_count      =  top[0] -> count();
    ins_args.pooled_height_ =  pooled_height_;
    ins_args.pooled_width_  =  pooled_width_;
    ins_args.width_         =  width_;
    ins_args.height_        =  height_;
    ins_args.channels_      =  channels_;
    ins_args.argmax_data    =  argmax_data;
    ins_args.spatial_scale_ =  spatial_scale_;
    int i;
    for (i = 0; i < MAX_CPU_NUMBER; i++)
    {    
        queue_Q[i].routine     = roi_pool_inner_thread;
        queue_Q[i].position    = i;
        queue_Q[i].args        = &ins_args;
    }

    all_sub_pthread_exec(queue_Q, MAX_CPU_NUMBER);
  
    clock_gettime(CLOCK_MONOTONIC, &finish);
    roipool_elapsed_time += (finish.tv_sec - start.tv_sec);
    roipool_elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
   // printf("roipool_elapsed_time:%f\r\n", roipool_elapsed_time);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
