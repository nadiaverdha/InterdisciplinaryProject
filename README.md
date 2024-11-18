# InterdisciplinaryProject
Repository for Interdisciplinary Project in DS at TU Wien


This project aims to develop a U-Net model for segmenting water bodies and monitoring the water extent of lake Neusiedl and its surrounding Lacken, using Sentinel-1 images.


#### Project Structure
- `data_func.py` & `data_prep.ipynb` - data preparation of my dataset for the model. Here the radar images are merged together, separate Lacken masks are created, and dataset is divided into train, validation and test set
- `model.py` - contains Encoder, Decoder and UNet classes
- `model.py` - contains an UNet model class which incorporates a pre-trained model
- `utils.py` - contains the necessary functions needed for training the model and for calculating the metrics
- `model_utils.py` - contains function for training, evaluating, saving best model
- `train&evaluate_model***.ipynb`- the notebooks 1 to 5 are used for training the different models and evaluating them on test set
- `inference.ipynb`  -  the results of all the models are shown in one notebook through plots
