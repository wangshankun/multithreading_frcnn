#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
#include "caffe/calcu_pthread.h"

#define savefile(name, buffer, size) do\
{\
  printf("############file size:%d############\r\n",size);\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


static int  scount;
static char fname[32];

typedef struct {
    float*  bottom_data;
    float*  top_data;
    int*    range_channel;
    float   negative_slope;
} relu_arg_t;

static void divide(int M, int* range_M)
{
    int dx = M%MAX_CPU_NUMBER;
    int dy = M/MAX_CPU_NUMBER;
    int index = 0;
    int i;
    for(i = 0;i < MAX_CPU_NUMBER + 1; i++)
    {
        range_M[i] = index;
        if(i < dx)
        {
            index = index + dy + 1;
        }
        else
        {
            index = index + dy;
        }
    }
}

void relu_inner_thread(void* set_args, int pos)
{
    relu_arg_t* args           = (relu_arg_t*)set_args;
    float*      bottom_data    = args -> bottom_data;
    float*      top_data       = args -> top_data;
    int*        range_channel  = args -> range_channel;
    float       negative_slope = args -> negative_slope;

    for (int i = range_channel[pos]; i < range_channel[pos + 1]; ++i) 
    {
        top_data[i] = std::max(bottom_data[i], (float)(0)) + negative_slope * std::min(bottom_data[i], (float)(0));
    }
}

namespace caffe {
double relu_elapsed_time=0;
template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
struct timespec start, finish;
clock_gettime(CLOCK_MONOTONIC, &start);
double sub_relu_elapsed_time = 0; 

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  
  
    int         range_channel[MAX_CPU_NUMBER + 1];
    queue_t     queue_Q[MAX_CPU_NUMBER];
    relu_arg_t  ins_args;

    divide(count, range_channel);
    ins_args.bottom_data    =  (float*)bottom_data;
    ins_args.top_data       =  (float*)top_data;
    ins_args.range_channel  =  range_channel;
    ins_args.negative_slope =  negative_slope;
    int i;
    for (i = 0; i < MAX_CPU_NUMBER; i++)
    {    
        queue_Q[i].routine     = relu_inner_thread;
        queue_Q[i].position    = i;
        queue_Q[i].args        = &ins_args;
    }
    all_sub_pthread_exec(queue_Q, MAX_CPU_NUMBER);
  
    clock_gettime(CLOCK_MONOTONIC, &finish);
    sub_relu_elapsed_time = (finish.tv_sec - start.tv_sec);
    sub_relu_elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    relu_elapsed_time += sub_relu_elapsed_time;
//sprintf(fname, "relu_%d", scount);
//savefile(fname, (float*)top_data, count*4 );
//scount++;
//    printf("relu_elapsed_time:%f sub_relu_elapsed_time:%f count:%d\r\n",relu_elapsed_time,sub_relu_elapsed_time,count);
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
