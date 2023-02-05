# Painter Impersonation using GANs

This repository contains notebooks and pythonf files for this [kaggle competition](https://www.kaggle.com/competitions/gan-getting-started/). 

The repository has the following structure:

```{shell}
.
├── DiffAugment.ipynb
├── cycleGAN.ipynb
├── dualdiscriminator.ipynb
├── requirements.txt
└── utils
    ├── cyclegan.py
    ├── diffaugmentation.py
    ├── dualdiscriminator.py
    ├── gan.py
    └── preprocessing.py
```

This code was developed on MacOS, Linux, and the Kaggle Jupyter notebok interface. The requirements for a MacOS with silicon chip are contained in the `requirements.txt` file. The Python version of the environment is 3.8.14. 

In the Kaggle interface, we recomment using the TPU v3-8, for faster training computations. 

## In a nutshell
This repository is composed of two main parts, the notebooks and the python code in the `utils` folder. 

For this project, we have tested 3 methods of GANs. The initial goal of the Kaggle competition was to generate 7,000 Monet-esque pictures. 

![image](https://user-images.githubusercontent.com/60437222/216825705-f26aa0d7-cc4b-4ebf-910b-a0defebf240d.png)

In order to do so, we have tested:
* a vanilla CycleGAN
* a vanilla CycleGAN + Differential Augmentation
* a vanilla CycleGAN + Differential Augmentation + Dual Headed Discriminator. 

The theory and report for this project can be found [here](https://www.notion.so/laurendu/Style-Transfer-Using-GANs-Painter-Impersonation-a226f81f209c4cefbf2bba7a2969d702). 
