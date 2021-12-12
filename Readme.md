# Multi-Modal Answer Validation for Knowledge-Based VQA
By Jialin Wu, Jiasen Lu, Ashish Sabharwal and Roozbeh Mottaghi

In this project, we present **M**ulti-modal **A**nswer **V**alidation using **Ex**ternal knowledge (MAVEx). The idea is to validate a set of promising answer candidates based on answer-specific knowledge retrieval. In particular, MAVEx aims to learn how to extract relevant knowledge from noisy sources, which knowledge source to trust for each answer candidate, and how to validate the candidate using that source.

## Installation
0. Requirements

   We implement this codebase on Ubuntu 18.04.5 LTS with TITAN V GPUs.

1. Clone this repository

   ```
   git clone git@github.com:jialinwu17/MAVEX.git
   ```
   
2. Using `conda`, create an environment
   As the implementation is based on ViLBERT-multi-task system, we require a similar virtual environment. Please refer to the Repository Setup step in [ViLBERT](https://github.com/facebookresearch/vilbert-multi-task) repository 
   
## Data Preparation
0. Object detection features and base ViLBERT pretrained model.

    As OK-VQA test set contains images that are used in both the object detection module that provides bottom-up attentions and the official released ViLBERT pretrained model, we carefully removed the OK-VQA test images from Visual Genome and COCO dataset and re-train the ResNeXT-152 based Faster RCNN object detector and then the  ViLBERT model from scratch following the default hyperparameters.
    
    The object features can be downloaded from [here](https://mavex.s3.us-east-2.amazonaws.com/new_image_features.zip). Aftr downloading it, please unzip it as 'image_features'
    
    The ViLBERT pretrained model can be downloaded from [here](https://mavex.s3.us-east-2.amazonaws.com/pytorch_model_4.bin)
    
1. Google Image features.

    We query Google Image search engine for the external visual knowledge and we process the retrieved images using the object detection module form the last step. Please download the processed image features and idx files following the instructions in below.
    (1) mkdir h5py_accumulate. <br>
    (2) download [train_idx](https://mavex.s3.us-east-2.amazonaws.com/h5py_accumulate/image_features/image_train_qid_ans2idx.pkl) to h5py_accumulate.<br>
    (3) download [train_features](https://mavex.s3.us-east-2.amazonaws.com/h5py_accumulate/image_features/image_train.hdf5) to h5py_accumulate.<br>
    (4) download [val_idx](https://mavex.s3.us-east-2.amazonaws.com/h5py_accumulate/image_features/image_val_qid_ans2idx.pkl) to h5py_accumulate.<br>
    (5) download [val_features](https://mavex.s3.us-east-2.amazonaws.com/h5py_accumulate/image_features/image_val.hdf5) to h5py_accumulate.<br>
    
2. Retrieved Knowledge

    Please download retrieved knowledge from [here](https://drive.google.com/file/d/1F_tKHOC5HIHdmm9KnUJV7wYMBZEgHQI0/view?usp=sharing)<br>
    
## Training
Train by runnning <br>
```
python ft_mavex.py --save_name demo --seed 7777 --from_pretrained pytorch_model_4.bin --num_epochs 75
```

## Models and Output files
We publish the MAVEx finetuned model at [here](http://www.cs.utexas.edu/~jialinwu/mavex.bin) and the output results can be downloaded [here](https://drive.google.com/drive/folders/1V4hgm1OXRvD7TlFADEXtUMHxKi6ePDc8?usp=sharing)

## Citation

If you find this project useful in your research, please consider citing our paper:

```
@inproceedings{khz2021interact,
  author = {Wu, Jialin and Lu, Jiasen and Sabharwal, Ashish and Mottaghi, Roozbeh},
  title = {{M}ulti-{M}odal {A}nswer {V}alidation for {K}nowledge-Based {VQA}},
  booktitle = {AAAI},	    
  year = {2022}
}
```
