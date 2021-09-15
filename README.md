# Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging
This repository is for the following paper:  
Zhuoyuan Wu, [Jian Zhang](https://jianzhang.tech/), Chong Mou; Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging; IEEE International Conference on Computer Vision (ICCV) 2021 [\[arxiv\]](https://arxiv.org/abs/2109.06548)  

Code is coming soon!  
## Requirements
- Python 3.6
- PyTorch >= 1.1.0
- numpy
- skimage
- cv2  
## Introduction  
Inspired by the half quadratic splitting (HQS) algorithm, we put forward a novel dense deep unfolding network with 3D-CNN prior for Snapshot compressive imaging. Merging the merits of both model-based methods and learning-based methods, our method has strong interpretability and high-quality reconstructed results. To enhance the ability to exploit spatial-temporal correlation, we assemble a deep network with 3D-CNN prior. To reduce the information loss, we propose a strategy of dense feature map (DFM) fusion, and we also design a dense feature map adaption (DFMA) module to make information optionally transmitting between phases.
![network](/Figs/network.PNG)
## Citation
If you find our work helpful in your resarch or work, please cite the following paper.
```
@inproceedings{mou2021gatir,
  title={Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging},
  author={Zhuoyuan, Wu and Jian, Zhang and Chong, Mou},
  booktitle={IEEE International Conference on Computer Vision},
  year={2021}
}
```
