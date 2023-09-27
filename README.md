# YOLO & Monodepth2
Real-time time motorcycle detection and range estimation with MDT-YOLOv4 and Monodepth in a single image 

This model is presented for specifically detecting motorcycles with black windshields.

This model was trained using a custom dataset from motorcycles in Tehran's streets.

## Result
![24](https://github.com/zahrabsh74/YOLOMonodepth/blob/main/results.png)


## Inference
- Download the Modepth_model [] and Yolo_model []

- Run the following code:
```
python final_test.py ---image_dire "./motorcycle_image"
```
