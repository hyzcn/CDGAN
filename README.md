# VRAR-Course-Project
This pro

# DeblurGAN
## Installation

```
virtualenv venv -p python3
. venv/bin/activate
pip install -r requirements.txt
```
## Dataset

Get the [GOPRO dataset](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing), and extract it in the `deblur-gan` directory. The directory name should be `GOPRO_Large`.


Use:
```
python organize_gopro_dataset.py --dir_in=GOPRO_Large --dir_out=images
```


## Training

```
python train.py --n_images=512 --batch_size=16
```

Use `python train.py --help` for all options

## Testing

```
python test.py
```

Use `python test.py --help` for all options

## Deblur your own image

```
python deblur_image.py --image_path=path/to/image
```
## Models



# Team Members
- [Zhiwen Qiang](https://github.com/QLightman)

- [Hao Wang](https://github.com/wrystal)

