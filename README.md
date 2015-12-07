## Two-Stream CNN

### Caffe models

Folder **./caffe_model/** contains the spatial and temporal caffe models of the two-stream CNN, pre-trained by [Limin Wang](https://github.com/wanglimin/TDD) on the split1 of UCF101 dataset. They used [dense_flow](https://github.com/wanglimin/dense_flow) to extract the optical flow of videos.

### MatConvNet models

In folder **./matconv_model/**, I have transformed the binary caffe models into matconvnet models.

### Input Format
* Caffe Model:
  * Spatial:  [W] x [H] x [B,G,R]
  * Temporal: [W] x [H] x [FlowX,FlowY]^10
* MatConvNet:
  * Spatial:  [W] x [H] x [B,G,R]
  * Temporal: [W] x [H] x [FlowX,FlowY]^10
