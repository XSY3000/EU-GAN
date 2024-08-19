# EU-GAN: A Root Inpainting Network for Improving 2D Soil-cultivated Root Phenotyping

This repository contains the code for this [paper]()

## installation requirements
- an NVIDIA GPU and a driver
- CUDA
- cuDNN
- python 3.9

## **installation**

1. Clone the repository
```
git clone https://github.com/XSY3000/EU-GAN.git
cd EU-GAN
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

## **usage**
The "train()", "test()" and "infer()" functions are defined in the "trainer.py".
"main.py" shows how to use the functions.
### **Infer**
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
```python
from load_config import Config
from trainer import Trainer
config = Config('configs/config1.yml')
trainer = Trainer(config)
trainer.train()
```
### **Test**
```python
from load_config import Config
from trainer import Trainer
config = Config('configs/config1.yml')
trainer = Trainer(config)
trainer.test(weight='weights/EU-GAN/bestmodel.pth')
```

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
