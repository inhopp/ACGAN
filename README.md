# ACGAN

Conditional image synthesis with auxiliary classifier GANs (AC-GANs)

> [Paper Review](https://inhopp.github.io/paper/Paper16/)

| Epoch 0 | Epoch 50 | Epoch 100 | Epoch 150 | Epoch 200 |
|:-:|:-:|:-:|:-:|:-:|
| ![data0](https://user-images.githubusercontent.com/96368476/215316520-03512d96-1d3b-4eae-b16a-30c7e042c5fc.png) | ![data49](https://user-images.githubusercontent.com/96368476/217025804-0e9fa183-8b8a-4c43-a02d-06bbf49fd0f8.png) | ![data99](https://user-images.githubusercontent.com/96368476/217025815-a405cfa8-c64e-4433-92e2-eff27f190cbf.png) | ![data149](https://user-images.githubusercontent.com/96368476/217025819-fdaab384-78fd-48de-9d88-e78bff90375f.png) | ![data199](https://user-images.githubusercontent.com/96368476/217025825-4d41735d-2082-486d-936b-610074ebe280.png) |


## Repository Directory 

``` python 
├── ACGAN
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
git clone https://github.com/inhopp/ACGAN.git
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