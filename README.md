# stereo-depth

GPU accelerated single view passive stereo depth estimation pipeline.

![Github Readme Diagram](https://user-images.githubusercontent.com/27950949/185786859-ee506e98-cece-4341-bdff-87c0ece321a1.png)

## Features
 * Real-time DNN based right view generation
 * Multiple depth estimation backends
    * Real-time CUDA stereo matching algorithm
    * Group-wise Correlation Stereo Network (GwC-Net)
    * MobileStereoNet (MSNet2D & MSNet3D)
 * REST API for the entire depth estimation pipeline


## Main references
 * Right view synthesis
    ```
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Deep3d: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks." European conference on computer vision. Springer, Cham, 2016
    ```
    ```
    Luo, Yue, et al. "Single view stereo matching." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    ```
 * CUDA stereo matching
    ```
    Chang, Qiong, and Tsutomu Maruyama. "Real-time stereo vision system: a multi-block matching on GPU." IEEE Access 6 (2018): 42030-42046.
    ```
 * GwC-Net
    ```
    Guo, Xiaoyang, et al. "Group-wise correlation stereo network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    ```
 * MobileStereoNet
    ```
    Shamsafar, Faranak, et al. "Mobilestereonet: Towards lightweight deep networks for stereo matching." Proceedings of the ieee/cvf winter conference on applications of computer vision. 2022.
    ```
