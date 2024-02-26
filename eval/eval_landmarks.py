import os
import numpy as np
from ultralytics import YOLO
import argparse
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np

def parse_args():
    """
    Parse command line arguments for evaluating landmarks.
    
    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate Landmarks')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--im_path', type=str, default=None, help='Path to the images')
    parser.add_argument('--lab_path', type=str, default=None, help='Path to the labels')
    parser.add_argument('--output_path', type=str, default='.', help='Path to output folder')
    parser.add_argument('--px_threshold', type=float, default=10, help='The maximum pixel mean pixel error for downselection')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='The confidence threshold for detection')
    parser.add_argument('--err_path', type=str, default=None, help='Path to the error file if previously computed')
    parser.add_argument('--calculate_err', action='store_true', help='Calculate the error')
    parser.add_argument('--save_err', action='store_true', help='Save the error')
    parser.add_argument('--best_classes', action='store_true', help='Calculate the best classes')
    parsed_args = parser.parse_args()

    if parsed_args.calculate_err and parsed_args.err_path is not None:
        raise ValueError('Cannot calculate error and provide error file at the same time')
    if parsed_args.calculate_err and parsed_args.lab_path is None:
        raise ValueError('Cannot calculate error without labels')
    if parsed_args.calculate_err and parsed_args.im_path is None:
        raise ValueError('Cannot calculate error without images')
    if parsed_args.calculate_err and parsed_args.model_path is None:
        raise ValueError('Cannot calculate error without model')
    
    

    return parsed_args

def load_model(model_path):
    """
    Load the model from the given file path.
    
    Args:
        model_path (str): The path to the model file.
        
    Returns:
        ultralytics.YOLO: The loaded model.
    """
    model = YOLO(model_path)
    return model

def get_err(err_path):
    """
    Load the error file from the given file path.
    
    Args:
        err_path (str): The path to the error file.
        
    Returns:
        numpy.ndarray: The loaded error file.
    """
    err = np.load(err_path)
    return err

def read_labels(lab_path):
    """
    Read the labels from the given file path.
    
    Args:
        lab_path (str): The path to the labels file.
        
    Returns:
        numpy.ndarray: The loaded labels.
    """
    with open(lab_path) as infile:
        labels = infile.readlines()
    if len(labels) > 0:
        labels = [x.split() for x in labels]
        labels = [[int(float(x[0])), float(x[1]), float(x[2])] for x in labels]
    else:
        return []
    return np.array(labels)

def get_dets():
    """
    Get the detections from the given model and image path.
    
    Args:
        model (ultralytics.YOLO): The model to use for inference.
        im_path (str): The path to the images.
        conf_threshold (float): The confidence threshold for detection.
        
    Returns:
        numpy.ndarray: The detections.
    """
    model = load_model(args.model_path)
    results = model(args.im_path, conf=args.conf_threshold)
    dets = []
    for result in results:
        if len(result.boxes) > 0:
            classes = result.boxes.cls
            confs = result.boxes.conf
            xcns = result.boxes.xyxyn[:, 0]
            ycns = result.boxes.xyxyn[:, 1]
            (im_h, im_w) = result.orig_shape
            im_hs = torch.ones_like(xcns) * im_h
            im_ws = torch.ones_like(ycns) * im_w
            im_dets = torch.stack([classes, xcns, ycns, im_ws, im_hs, confs], dim=1)
            dets.append(im_dets)
        else:
            dets.append([])
    return dets

def calculate_error(labels, dets):
    """
    Calculate the error between the detections and the labels.
    
    Args:
        labels (list): The labels.
        dets (list): The detections.
        
    Returns:
        numpy.ndarray: The error.
    """
    err_list = []
    for label, det in zip(labels, dets):
        if len(det) > 0:
            print(len(det))
            if len(label) > 0:
                print(len(label))
                label_classes = label[:, 0]
                label_xcs = label[:, 1]
                label_ycs = label[:, 2]
                for det_cl, det_xc, det_yc, det_im_w, det_im_h, det_conf in det:
                    det_cl = int(det_cl)
                    if det_cl in label_classes:
                        label_idx = np.where(label_classes == det_cl)[0][0]
                        label_x = label_xcs[label_idx]
                        label_y = label_ycs[label_idx]
                        xerr = abs(det_xc - label_x) * det_im_w
                        yerr = abs(det_yc - label_y) * det_im_h
                        err = np.sqrt(xerr ** 2 + yerr ** 2)
                        err_list.append([det_cl, err, det_conf])
                    else:
                        err_list.append([det_cl, -1, det_conf])
            else:
                for det_cl, det_xc, det_yc, det_im_w, det_im_h, det_conf in det:
                    err_list.append([det_cl, -1, det_conf])
        else:
            if len(label) > 0:
                label_classes = label[:, 0]
                for label_cl, label_x, label_y in label:
                    err_list.append([label_cl, -1, -1])
    err_arr = np.array(err_list)
    if args.save_err:
        np.save(os.path.join(args.output_path, 'err.npy'), err_arr)
        print('Error saved to {}'.format(os.path.join(args.output_path, 'err.npy')))
    return np.array(err_arr)

def get_class_values(err):
    """
    Get the class values from the error file.
    
    Args:
        err (numpy.ndarray): The error file.
        
    Returns:
        numpy.ndarray: The class values.
    """
    classes = np.unique(err[:, 0])
    return classes

def calculate_class_stats(err, cl):
    """
    Calculate the mean error, median error, mean confidence, missed detections, and extra detections for the given class.
    
    Args:
        err (numpy.ndarray): The error file.
        cl (int): The class to calculate the statistics for.
        
    Returns:
        float: The mean error.
        float: The median error.
        float: The mean confidence.
        int: The number of missed detections.
        int: The number of extra detections.
    """
    cl_errs = err[err[:, 0] == cl]
    cl_errs = cl_errs[cl_errs[:, -1] > args.conf_threshold]
    mean_err = np.mean(cl_errs[cl_errs[:,1] > 0, 1])
    median_err = np.median(cl_errs[cl_errs[:,1] > 0, 1])
    mean_conf = np.mean(cl_errs[cl_errs[:,1] > 0, 2])
    missed = np.sum(cl_errs[:, 2] == -1)
    extra = np.sum(cl_errs[:, 1] == -1)
    return cl, mean_err, median_err, mean_conf, missed, extra

args = parse_args()

if __name__ == '__main__':
    if args.calculate_err:
        lab_folder = args.lab_path
        lab_paths = os.listdir(lab_folder)
        lab_paths = [os.path.join(lab_folder, x) for x in lab_paths]
        labels = [read_labels(lab_path) for lab_path in lab_paths] 
        dets = get_dets()
        err = calculate_error(labels, dets)
    elif(args.err_path is not None):
        err = get_err(args.err_path)
    else:
        print('No error file provided or error calculation requested')
    if args.best_classes:
        classes = get_class_values(err)
        class_stats = [calculate_class_stats(err, cl) for cl in classes]
        class_stats = np.array(class_stats)
        class_stats = class_stats[class_stats[:, 0].argsort()]
        print('Class, Mean Error, Median Error, Mean Confidence, Missed Detections, Extra Detections')
        for cl, mean_err, median_err, mean_conf, missed, extra in class_stats:
            print('{}, {:.2f}, {:.2f}, {:.2f}, {}, {}'.format(int(cl), mean_err, median_err, mean_conf, int(missed), int(extra)))