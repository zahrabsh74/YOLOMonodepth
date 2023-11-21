# YOLO & Monodepth2
Real-time time motorcycle detection and range estimation with MD-TinyYOLOv4 and Monodepth in a single image 

This model is presented for specifically detecting motorcycles with black windshields.

This model was trained using a custom dataset from motorcycles in Tehran's streets.
## Requirements
- Tensorflow
- Opencv
- torch
- torchvision
- matplotlib

## Dataset
the dataset is available in [data](https://drive.google.com/drive/folders/1ZmSOmuEaLK_kwa5fSUrffk7FnxaQrXTr?usp=drive_link) 
## Result
![24](https://github.com/zahrabsh74/YOLOMonodepth/blob/main/results.png)


## Inference
- Download the [Modepth_model](https://drive.google.com/file/d/1LjhElUvirdqLEJgn84tK7hK-mhZmeKsZ/view?usp=drive_link) and [Yolo_model](https://drive.google.com/file/d/1SMbPmBQF1t_pd6iPSnfBXWxNw5AxNLPV/view?usp=drive_link)

- Run the following code:
```
python final_test.py ---image_dire "./motorcycle_image"
```
