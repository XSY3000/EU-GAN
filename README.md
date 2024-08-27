# EU-GAN: A Root Inpainting Network for Improving 2D Soil-cultivated Root Phenotyping

This repository contains the code for this [paper]()

## Abstract
The root system is crucial for nutrient absorption and plant stability, playing a significant role in crop growth and resilience. Rhizobox, as a cost-effective root imaging method, enables in situ and non-destructive phenotypic detection of roots in soil. However, the opacity of the soil often leads to intermittent gaps in the root images, which in turn reduces the accuracy of root phenotype calculations. In this paper, we present a novel root inpainting method based on GAN architecture, specifically designed to overcome the limitations of existing approaches. Our method avoids cropping during training by instead utilizing downsampled images to provide the overall root morphology. By employing Binary Cross-Entropy Loss and Dice Loss, the model is trained to focus more on the missing root pixels. Additionally, we remove the skip connections in U-Net and introduce an Edge Attention Module (EAM) to capture more detailed information. Compared to other methods, our approach significantly **improves the recall rate from 17.35% to 35.75%** on a dataset of 122 cotton root images, demonstrating enhanced inpainting capabilities. The mean absolute error for root area, root length, convex hull area, root width, and root depth decreased by **76.07%, 68.63%, 48.64%, 1.32%, and 88.28%**, respectively, leading to a substantial improvement in root phenotyping accuracy. This method provides a robust tool for high-throughput 2D soil-cultivated root phenotyping, thereby enhancing breeding program efficiency and advancing our understanding of root system dynamics.

## Installation Requirements
- an NVIDIA GPU and a driver
- CUDA
- cuDNN
- python 3.9

## **Installation**

1. Clone the repository  

Git:
```
git clone https://github.com/XSY3000/EU-GAN.git
cd EU-GAN
```
zip:
```
download this repository's zip file
unzip EU-GAN-main.zip
cd EU-GAN-main
```
2. Install the requirements
```
pip install -r requirements.txt
```
3. Download the dataset and weights
>The dataset and weights can be downloaded from [here](https://drive.google.com/drive/folders/1H9u0tMWM6EqAU3g0LItMFJefoRmFgYN1?usp=drive_link) 
> 
>Put the "EU-GAN" directory in the "weights" directory.  
>Unzip the "filedata.zip" file to the root directory.

## Usage
The "train()", "test()" and "infer()" functions are defined in the `trainer.py`.  

`main.py` shows how to use these functions.
### **Infer**
To perform inference, use the `infer()` function as shown below:
```python
from load_config import Config
from trainer import Trainer
image_path = 'your/image/path'
save_path = 'your/save/path' # if '', the result will be saved in the log directory.
config = Config('configs/config1.yml')
trainer = Trainer(config)
trainer.infer(image_path, save_path, weights='weights/EU-GAN/bestmodel.pth')
```
### **Train**
To train the model, use the train() function:
```python
from load_config import Config
from trainer import Trainer
config = Config('configs/config1.yml')
trainer = Trainer(config)
trainer.train()
```
check the `configs/config1.yml` to change the hyperparameters.
### **Test**
To test the model, use the `test()` function:
```python
from load_config import Config
from trainer import Trainer
config = Config('configs/config1.yml')
trainer = Trainer(config)
trainer.test(weight='weights/EU-GAN/bestmodel.pth')
```

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
