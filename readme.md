# ConvCRF.pytorch

This repository is an unofficial pytorch implementation of 
[Convolutional CRFs for Semantic Segmentation](https://arxiv.org/abs/1805.04777).
We refer to [the official code](https://github.com/MarvinTeichmann/ConvCRF) for our version, the difference between us is
- We rebuild the core code and add enough comments to make it easier to understand
- We add training code [will come soon]
- We build ConvCRF3D to process 3D images (includes CT images)

## Requirements
- Pytorch>=0.4.0
- CPU or GPU
- Other packages can be installed with the following instruction:
```
pip install requirements.txt
```
  
## Quick start
Running the code with the following command.
```
python demo2d.py
```
```buildoutcfg
python demo3d.py
```
Note: You can modify some parameters in "configs/config2d.py" or "configs/config3d.py" to get your own specific models.

## Citation
```
@article{Teichmann2018Convolutional,
  title={Convolutional CRFs for Semantic Segmentation},
  author={Teichmann, Marvin T. T and Cipolla, Roberto},
  year={2018},
}
```