### Purpose: Use Residual block and create a CNN model for training on CiFAR 10 dataset.

## Based on CiFAR 10 and MNIST datasets
### Details:- 
<p>
First part of the assignment is to train your own UNet from scratch, you can use the dataset and strategy provided in this linkLinks to an external site.. However, you need to train it 4 times:

MP+Tr+BCE
MP+Tr+Dice Loss
StrConv+Tr+BCE
StrConv+Ups+Dice Loss
</p>

Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(16, 16) <br/>
Batch size = 512 <br/>
Use SGD, and CrossEntropyLoss <br/>

### Project Setup:
Clone the project as shown below:-

```bash
$ git clone git@github.com:pankaja0285/era_v1_session18.git
```
About the file structure</br>
|__session18_bce_dice_unet_segmentation.ipynb
|__session18_VAE_PL.ipynb<br/>
|__README.md<br/>

**NOTE:** List of libraries required: 
- ***torch*** 
- ***torchvision***
- ***lightning-bolts***<br/>
***lightning-bolts*** - installs all the dependencies like the
pytorch-lightning (1.9.5) and lightning-bolts (0.7.0)
Install as

!pip install torch torchvision <br/>
!pip install lightning-bolts

One of 2 ways to run any of the notebooks, for instance **Submission_CiFAR_S11_GradCam.ipynb** notebook:<br/>
1. Using Anaconda prompt - Run as an **administrator** start jupyter notebook from the folder ***era_v1_session11_pankaja*** and run it off of your localhost<br/>
**NOTE:** Without Admin privileges, the installs will not be correct and further import libraries will fail. <br/>
```
jupyter notebook
```
2. Upload the notebook folder ***era_v1_session11_pankaja*** to google colab at [colab.google.com](https://colab.research.google.com/) and run it on colab<br/>

### In <i>session18_bce_dice_unet_segmentation.ipynb</i> - Use case With RandomCrop ONLY:

<p>
Target:
- To train a UNET model from scratch and apply BCE And DICE Loss

Results:
- Total Trainable parameters: 174 M
- Train loss as low as possible 

Analysis:
- To see how the tranformers work and the affects of the DICE, BCE 
  respectively.
</p>

### In <i>session18_VAE_PL.ipynb</i> - Here Variational Auto Encoders is implemented

<p>
Target:
- Using a Resnet18 encoder / decoder and see how the 
  transformers behave

Results:
- Total parameters: 20.1 M
- Train loss as low as possible  

Analysis:
- To see how the the tranformers work based on VAE
</p>

### Contributing: This is a combined effort by 
Divya GK: @gkdivya <br/>
Pankaja Shankar: @pankaja0285<br/>

For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
