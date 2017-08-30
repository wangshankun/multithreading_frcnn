### code form
- org code from  https://github.com/rbgirshick/py-faster-rcnn.git

### Add fucntion
- add multithreading for caffe:

    modified:   src/caffe/layers/pooling_layer.cpp

    modified:   src/caffe/layers/relu_layer.cpp

    modified:   src/caffe/layers/roi_pooling_layer.cpp

    modified:   src/caffe/net.cpp

    modified:   src/caffe/util/im2col.cpp

    add:        include/caffe/calcu_pthread.h

    add:        src/caffe/calcu_pthread.cpp

    add:        Makefile.config


- add westwell port project using modle:

    modified:   tools/demo.py

    add:        westwell/


###compile:

- cd multithreading_frcnn/caffe-fast-rcnn

  make -j8 && make pycaffe

- cd multithreading_frcnn/lib

  make

###exec:
-  ./tools/demo.py --cpu

