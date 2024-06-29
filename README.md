# Image Aesthetics Assesment

## Usage

### 1. Colab directory:
- https://drive.google.com/drive/folders/1kdVjFMXKzBMCwCPnb63Y6dUhmuWRpk6O?usp=drive_link
- 确保路径和colab保持一致

### 2. Data:
- TAD66K
- 用merge的label


### 3. Train:
```sh
python IAA/train.py --epochs 2 --csv_file TAD66K/label/cleaned_train.csv --img_dir TAD66K/image --output_model_path exp/model/IAA_2.pth
