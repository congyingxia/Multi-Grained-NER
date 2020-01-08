# Multi-Grained Named Entity Recognition


This repository implements a Multi-Grained Named Entity Recognition model proposed in this paper with Tensorflow:

Congying Xia, Chenwei Zhang, Tao Yang, Yaliang Li, Nan Du, Xian Wu, Wei Fan, Fenglong Ma, Philip Yu. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. 

https://arxiv.org/abs/1906.08449

# Requirements

Python

Tensorflow 1.13.1

AllenNLP

Numpy

Sklearn


# Usage

## Step 0.0, Prepare Data

 ### Get Datasets.
 
 The dataset content is subject to copyright issues. Some useful pointers:

    (ACE 2004) https://catalog.ldc.upenn.edu/LDC2005T09

    (ACE 2005) https://catalog.ldc.upenn.edu/LDC2006T06

    (CoNLL 2003) https://cogcomp.seas.upenn.edu/page/resource_view/81


 ### Preprocess the datasets into required data formats.

   Two types of format are needed. The first example can be found at ***data/ace2004/train.txt***. The second examples can be found at ***data/ace2004/elmo/sentences/train_sentences***.
     
   Please generate formated files for the whole datasets and replace the example files with your generated files, including:
    
    data/ace2004/train.txt
    
    data/ace2004/test.txt
    
    data/ace2004/dev.txt
    
    data/ace2004/elmo/sentences/train_sentences
    
    data/ace2004/elmo/sentences/test_sentences
    
    data/ace2004/elmo/sentences/dev_sentences


### Word embeddings: 

   Download ***glove.6B.zip*** from https://nlp.stanford.edu/projects/glove/. 

   Unzip it and put ***glove.6B.300d.txt*** under the directory of ***data/glove.6B/***

### Elmo embeddings: 

Run the script to download Elmo embeddings.

    cd data/ace2004/elmo
    sh run.sh
  
 Three files should be generated:
 
    elmo_dev.hdf5
 
    elmo_test.hdf5
 
    elmo_train.hdf5 
 
 
## Step 1.0, Run Detector

  ### Step 1.1, Build data for the detector

  ```
  cd detector
  python build_data.py
  ```
  
  It should generate files including
        
    data/glove.6B.300d.trimmed.npz
    
    data/chars.txt
    
    data/tags.txt
    
    data/words.txt
          

   ### Step 1.2, train detector
 
  ```
  python train.py
  ```

   ### Step 1.3, dump shared features
  
  ```
  python dump.py
  ```
  Dumped Features will be saved in ***data/saved_roi/***

  If you'd like to evaluate the performance of the Detector, please run the evaluate.py
  
  ```
  python evaluate.py
  ```

## Step 2.0, Run Classifier

   ### Step 2.1, train classifier
  
  ```
  python train.py
  ```

   ### Step 2.2, evaluate model performance
  
  ```
  python evaluate.py
  ```

# Reference

If you find our code useful, please cite our paper.

```
@article{xia2019multi,
  title={Multi-Grained Named Entity Recognition},
  author={Xia, Congying and Zhang, Chenwei and Yang, Tao and Li, Yaliang and Du, Nan and Wu, Xian and Fan, Wei and Ma, Fenglong and Yu, Philip},
  journal={arXiv preprint arXiv:1906.08449},
  year={2019}
}

```
