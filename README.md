# CGAN

Conditional generative adversarial networks

> 어째 0이 조금 이상하다..

| Epoch 0 | Epoch 50 | Epoch 100 | Epoch 150 | Epoch 200 |
|:-:|:-:|:-:|:-:|:-:|
| ![data0](https://user-images.githubusercontent.com/96368476/215316520-03512d96-1d3b-4eae-b16a-30c7e042c5fc.png) | ![data49](https://user-images.githubusercontent.com/96368476/216553710-0e8a780c-90b8-4186-bd5a-484578408505.png) | ![data99](https://user-images.githubusercontent.com/96368476/216553706-a268af94-9024-4384-b681-7983432341cf.png) | ![data149](https://user-images.githubusercontent.com/96368476/216553703-4cb7b39b-6b76-4471-9f09-59ebe5da6c09.png) | ![data199](https://user-images.githubusercontent.com/96368476/216553699-c5c7d3e9-8cc2-405c-86cc-a98495c8b7b9.png) |


## Repository Directory 

``` python 
├── CGAN
     ├── datasets
     │     └── mnist
     ├── data.py
     ├── option.py
     ├── model.py
     ├── train.py
     └── README.md
```

- `data.py` : data load (download mnist)
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/CGAN.git
```

<br>


### train
``` python
python3 train.py
    --device {}(default: cpu) \
    --input_size{}(default: 32) \
    --n_classes{}(default: 10) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 200) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 64) \
```


<br>


#### Main Reference
https://github.com/eriklindernoren/PyTorch-GAN