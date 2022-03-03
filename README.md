# NRNM: Non-local-Recurrent-Neural-Memory
Official pytorch codes for two papers:  
[Non-local Recurrent Neural Memory for Supervised Sequence Modeling (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Non-Local_Recurrent_Neural_Memory_for_Supervised_Sequence_Modeling_ICCV_2019_paper.pdf), and
`Learning Sequence Representations by Non-local Recurrent Neural Memory (Being organized)`.

# Installation
The model is built in PyTorch 1.2.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).
You can installation the environment via the following:
```
pip install -r requirements.txt
```

# Training NRNM on NTU-Skeleton dataset
Run `Bash train_x.sh` to train LSTM-NRNM.

# Citation
If you find this work useful for your research, please cite:
```
@inproceedings{fu2019non,
  title={Non-local recurrent neural memory for supervised sequence modeling},
  author={Fu, Canmiao and Pei, Wenjie and Cao, Qiong and Zhang, Chaopeng and Zhao, Yong and Shen, Xiaoyong and Tai, Yu-Wing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6311--6320},
  year={2019}
}
```
