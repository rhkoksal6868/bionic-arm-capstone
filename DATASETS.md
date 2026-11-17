# Dataset notes

## HandGesture-5G

The source images used for `gesture_image_data.csv` are from [HandGesture-5G](https://data.mendeley.com/datasets/wj25crzdbm/1), licensed under CC BY 4.0. The source dataset is retained under `gesture_images/HandGesture-5G/` and tracked with Git LFS because it is about 4.1 GB. Keep the dataset attribution and license with any redistribution.

## HandGesture-Recognizer supplement

The additional 200 images under `gesture_images/HandGesture-Recognizer/` are excluded from version control until the image-level redistribution terms are confirmed. The repository's software license alone does not establish the images' license. The trained model and processed feature CSV remain included.

## Getting the files

Install Git LFS before cloning this repository so the source images are downloaded:

```bash
git lfs install
git clone <repository-url>
```

The `.venv/`, Python cache, and local `face_data/` enrollment samples are intentionally excluded. Face samples are personal biometric data and should be collected and shared only with each person's consent.
