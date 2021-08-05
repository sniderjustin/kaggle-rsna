# RSNA Kaggle Competition

See the competition online [here](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification).

## Setup

Clone and enter repo
```
git clone git@github.com:sniderjustin/kaggle-rsna.git
cd kaggle-rsna
```

Create and activate virtual environemnt
```
python -m venv venv/
source venv/bin/activate
```

Upgrade pip and install packages 
```
pip install -U pip
pip install -r requirements.txt
```

Create directory for tensorboard and launch
```
mkdir runs
tnesorboard --logdir=runs
```

Run the training proces
```
python sketch.py
```

## References

- [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) is the PyTorch model used. 
- The EfficientNet paper is [here](https://arxiv.org/abs/1905.11946).
- [This notebook](https://www.kaggle.com/blade001/brain-tumor-eda-with-score) by Kaggle Expert blade001 was used as a starting point. 