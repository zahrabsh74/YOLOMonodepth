
#================================================================
#   File name   : final_test.py
#   Author      : Zahrabsh74
#   Created date: 2020
#   Description :define functions of yolo detection and monodepth2 estimationin a combine_model
#                save output results of detection, distances and histograms
#                calculate time for each steps and also totall
#================================================================
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import networks
from networks.layers import disp_to_depth
import networks.yolo_utils as utils
import tensorflow as tf
import glob as glob
import time
import os
import math
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

###### combination function of Yolo $ monodepth
def combined_model(args):
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    num_classes= args.num_classes
    #INPUT_SIZE=320, 352, 384, "416", 448, 480, "512", 544, 576, 608
    input_size= args.INPUT_SIZE
    #make a data structures(graph) that contain a set of objects
    graph= tf.Graph()
    # Read Yolo pb tensors from utils.py
    return_tensors = utils.read_pb_return_tensors(graph, args.pb_file, return_elements)
    #images path direction
    path= glob.glob(os.path.join(args.image_dire+"/*.jpg"))
    path= glob.glob("./result_image/*.jpg")

    #using "cuda" or "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print("CUDA iS USED")
    else:
        device = torch.device("cpu")
        print("CPU iS USED")

##################### loding Monodepth Depth model
# ===================================================================================================//
# Copyright Niantic 2019. Patent Pending. All rights reserved.
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
#   the time of starting monodepth

    # Monodepth Finetuned model path direction
    model_path = os.path.join("./data/" + args.Modepth_model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    # Loading pretrained encoder
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # Loading pretrained decoder
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
# =======================================================================================\\

    #try to not occupied the whole gpu memory when Tensorflow is used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

####################  run YOLO and Monodepth2
    with tf.Session(graph=graph) as sess,torch.no_grad():

    ####################  run combined model for each image
        for image_path in path:

            #the time of starting for each image
            start = time.time()

            #read input image
            image_name = os.path.basename(image_path)
            original_image = cv2.imread(image_path)
            original_image_size = original_image.shape[:2]

            #the time of starting YOLO
            start_yolo = time.time()
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            #predict 3 scale predicted box from image by Yolo
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})
            # Join 3 predicted box along an existing axis, bboxes: (xmin, ymin, xmax, ymax, score, class)
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            #SCORE_THRESHOLD= 0.3
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, args.yolo_score_trsh)
            #IOU_THRESHOLD= 0.45
            bboxes = utils.nms(bboxes, args.yolo_IOU_trsh, method='nms')
            #drowing box and write class name detected motorcycle
            original_image,coord= utils.draw_bbox(original_image, bboxes,args.cfg)
            #the time of finishing YOLO
            end_yolo_time=time.time()

########################### depth of each image
#===================================================================================================//
# Copyright Niantic 2019. Patent Pending. All rights reserved.
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
            #the time of starting monodepth
            start_depth_time = time.time()

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            scaled_disparity, scaled_depth = disp_to_depth(disp_resized, 0.1, 100)
            disp_resized_np= scaled_disparity.squeeze().cpu().numpy()
            depth_resized_np= scaled_depth.squeeze().cpu().numpy()
# =================================================================================================\\
            #convert scaled_depth to real_distance with baseline=12 cm
            baseline=12
            STEREO_SCALE_FACTOR=baseline / 0.1
            real_distance=depth_resized_np*STEREO_SCALE_FACTOR


            #the time of ending monodepth
            end_depth_time = time.time()

            ##combine the results of YOLO and monodepth
            x_min = coord[0]
            y_min = coord[1]
            x_max = coord[2]
            y_max = coord[3]
            c1, c2 = (x_min, y_min), (x_max, y_max)
            center_x=math.floor(x_min+((x_max-x_min)/2))
            center_y=math.floor(y_min+((y_max-y_min)/2))
            kernel_distance=real_distance[center_y-2:center_y+3,center_x-2:center_x+3]

            center_point_distance=np.mean(real_distance[center_y-2:center_y+3,center_x-2:center_x+3])
            croped_box=real_distance[y_min:y_max,x_min:x_max]
            histo=cv2.calcHist([croped_box],[0],None,[1501],[0,1500])
            most_frequent=np.argmax(histo)
            print('min_distance=', croped_box.min(),'cm')
            print('max_distance=', croped_box.max(),'cm')
            print(image_name,'center_distance=',center_point_distance,"cm,",'most_frequent_distance=',most_frequent,"cm,",'kernel=',kernel_distance)

            #warn for motorcycles who are closer than 4meters
            if center_point_distance<400:
                print("!!!!!!!WARNING,the motorcycle is too close!!!!!!!!")
            end_yolodepth = time.time()

            ## Saving the yolo detection image
            original_image=pil.fromarray(original_image)
            original_image.save(os.path.join(args.image_dire + "/detection/detection_"+image_name))

            ### Saving histogram output
            plt.figure()
            plt.plot(histo)
            name_hist_im = os.path.join(args.image_dire+"/detection/hist_"+image_name)
            plt.savefig(name_hist_im)

            ## saving colormapped depth image
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            label='D=%scm' % (math.floor(center_point_distance))
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            cv2.rectangle(colormapped_im, c1, c2, [0, 0 ,0],2)
            cv2.putText(colormapped_im, label, (c1[0]+15, c1[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.5, [0, 0 ,0], thickness=1, lineType=cv2.LINE_AA)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(args.image_dire+"/detection/depth"+image_name)
            im.save(name_dest_im)
            end_output_saving = time.time()

            ## saving refine depth image by bilaterial filter
	    # Apply bilateral filter with d = 15,
	    # sigmaColor = sigmaSpace = 75.
            colormapped_bilateral = cv2.bilateralFilter(colormapped_im, 30, 85, 85)
            name_dest_bilateral = os.path.join(args.image_dire+"/detection/bilateral"+image_name)
            im2 = pil.fromarray(colormapped_bilateral)
            im2.save(name_dest_bilateral)
            end_output_saving = time.time()

   

            ### each image time-processing
            yolo_time = end_yolo_time-start_yolo
            print("yolo_time=",yolo_time)
            depth_time = end_depth_time-start_depth_time
            print("depth_time=",depth_time)
            yolodepth_time = end_yolodepth - start
            print("yolodepth_time=",yolodepth_time)
            totall_time_saving_output = end_output_saving - start
            print("totall_time_saving_output=",totall_time_saving_output)
            print('*' * 30)







