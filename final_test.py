
#================================================================
#   Copyright (C) 2023 * Ltd. All rights reserved.
#   File name   : final_test.py
#   Author      : Zahrabsh74
#   Created date: 2020
#   Description :take the input arguments for running the yolo and monodepth2 functions
#                for motorcycles detection and range estimation
#================================================================
from __future__ import absolute_import, division, print_function
from yolodepth import combined_model
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepth2 models.')
    #Monodepth Finetuned model path direction
    parser.add_argument('--Modepth_model_name', type=str,
                        help='name of a finetuned Monodepth model to use',default="finetuned2_stereo1024")
    #images path direction
    parser.add_argument('--image_dire', type=str,
                       help='image extension to search for in folder', default="./result_image")
    #using "cuda" or "cpu"
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    # Read Fietuned Yolo model parameter,
    parser.add_argument('--pb_file', type=str,
                         default="./data/yolo_test50.pb")
    parser.add_argument('--cfg', type=str,
                         default="./data/class.names")
    # SCORE_THRESHOLD= 0.3
    parser.add_argument('--yolo_score_trsh', type=float,
                         default="0.3")
    # IOU_THRESHOLD= 0.45
    parser.add_argument('--yolo_IOU_trsh', type=float,
                         default="0.45")
    #INPUT_SIZE=320, 352, 384, "416", 448, 480, "512", 544, 576, 608
    parser.add_argument('--INPUT_SIZE', type=int,
                         default="416")
    # detecting one motorcycle class
    parser.add_argument('--num_classes', type=int,
                         default="80")
    return parser.parse_args()



if __name__ == "__main__":


    args = parse_args()  
    combined_model(args)



 
