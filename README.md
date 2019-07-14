# Multi-Grained Named Entity Recognition


This repository implements a Multi-Grained Named Entity Recognition model proposed in this paper with Tensorflow:

Congying Xia, Chenwei Zhang, Tao Yang, Yaliang Li, Nan Du, Xian Wu, Wei Fan, Fenglong Ma, Philip Yu. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. 

https://arxiv.org/abs/1906.08449

# Usage


Step 1.0, Enter the detector directory and follow the README file

  Step 1.1, data preprocessing

  ```
  python build_data.py
  ```

  Step 1.2, train detector
  
  ```
  python train.py
  ```

  Step 1.3, dump shared features
  
  ```
  python dump.py
  ```

  If you'd like to evaluate the performance of the Detector, please run the evaluate.py
  
  ```
  python evaluate.py
  ```

Step 2.0, Enter the classifier directory and follow the README file

  Step 2.1, train classifier
  
  ```
  python train.py
  ```

  Step 2.2, evaluate model performance
  
  ```
  python evaluate.py
  ```

If you find our code useful, please cite our paper.

```
@article{xia2019multi,
  title={Multi-Grained Named Entity Recognition},
  author={Xia, Congying and Zhang, Chenwei and Yang, Tao and Li, Yaliang and Du, Nan and Wu, Xian and Fan, Wei and Ma, Fenglong and Yu, Philip},
  journal={arXiv preprint arXiv:1906.08449},
  year={2019}
}

```
