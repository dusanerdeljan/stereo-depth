# stereo-depth

GPU accelerated single view passive stereo depth estimation pipeline.

![Github Readme Diagram](https://user-images.githubusercontent.com/27950949/185786859-ee506e98-cece-4341-bdff-87c0ece321a1.png)

## Features
 * Real-time DNN based right view generation
 * Multiple depth estimation backends
    * Real-time CUDA stereo matching algorithm
    * Group-wise Correlation Stereo Network (GwcNet)
    * MobileStereoNet (MSNet2D & MSNet3D)
 * REST API for the entire depth estimation pipeline

## Results

### Right View Synthesis + CUDA stereo matching algorithm

https://user-images.githubusercontent.com/27950949/191323250-d241a2f9-9e64-45a9-bbc3-1a12966956f0.mp4

### Right View Synthesis + GwcNet

https://user-images.githubusercontent.com/27950949/191323286-15858ada-66a7-4b01-b4f7-29e5fe3bbb65.mp4

### Right View Synthesis + MobileStereoNet

https://user-images.githubusercontent.com/27950949/191323318-8a51019e-e17e-44b9-b967-a0965560ef27.mp4

## Main references
 * Right view synthesis
 
   * Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Deep3d: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks." European conference on computer vision. Springer, Cham, 2016

   * Luo, Yue, et al. "Single view stereo matching." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    
 * CUDA stereo matching
    
   * Chang, Qiong, and Tsutomu Maruyama. "Real-time stereo vision system: a multi-block matching on GPU." IEEE Access 6 (2018): 42030-42046.
   
 * GwcNet
    
   * Guo, Xiaoyang, et al. "Group-wise correlation stereo network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    
 * MobileStereoNet
    
   * Shamsafar, Faranak, et al. "Mobilestereonet: Towards lightweight deep networks for stereo matching." Proceedings of the ieee/cvf winter conference on applications of computer vision. 2022.
    
