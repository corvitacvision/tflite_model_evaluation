import time
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import xml.etree.ElementTree as Et

import multiprocessing

from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields
from object_detection.utils.label_map_util import create_categories_from_labelmap, get_label_map_dict


"""Input Parameters"""

nms = 0
data_dir = "evaluation/images/"     # Path to your label directory where x.jpg and x.xml 
base_eval_dir = "evaluation"        #  Base directory where you want to save your predictions and groundtruth labels together
model_file = "model.tflite"			# path to traine model
label_file = "label_map.pbtxt"		# path to label_map.pbtxt
input_mean = 128					# Used for quantization				
input_std = 128						# Used for quantization
score_threshold = 0.4				# Score threhold to remove false positive detections

interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def prepare_input(image_path):
	""" Input image preprocessing for SSD MobileNet format
	args:
		image_path: path to image
	returns:
		input_data: numpy array of shape (1, width, height, channel) after preprocessing
	"""
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]

	# Using OpenCV
	img = cv2.resize(cv2.imread(image_path), (width,height))

	input_shape = input_details[0]['shape']
	input_data = np.zeros(input_shape, dtype=np.uint8)
	input_data[0, :, :, :] = cv2.resize(img, (input_shape[2], input_shape[1]))
	return input_data


def postprocess_output(image_path):
	""" Output post processing
	args:
		image_path: path to image
	returns:
		boxes: numpy array (num_det, 4) of boundary boxes at image scale
		classes: numpy array (num_det) of class index
		scores: numpy array (num_det) of scores
		num_det: (int) the number of detections
	"""
	# SSD Mobilenet tflite model returns 10 boxes by default.
	# Use the output tensor at 4th index to get the number of valid boxes
	num_det = int(interpreter.get_tensor(output_details[3]['index']))
	boxes = interpreter.get_tensor(output_details[0]['index'])[0][:num_det]
	classes = interpreter.get_tensor(output_details[1]['index'])[0][:num_det]
	scores = interpreter.get_tensor(output_details[2]['index'])[0][:num_det]

	filtered_boxes = list()
	filtered_classes = list()
	filtered_scores = list()

	for box,label,score in zip(boxes,classes,scores):
		if score > score_threshold:
			filtered_boxes.append(box)
			filtered_classes.append(label)
			filtered_scores.append(score)

	filtered_boxes = np.array(filtered_boxes)
	filtered_classes = np.array(filtered_classes)
	filtered_scores = np.array(filtered_scores)

	# Scale the output to the input image size
	img_width, img_height = Image.open(image_path).size # PIL
	# img_height, img_width, _ = cv2.imread(image_path).shape # OpenCV

	num_det = len(filtered_scores)
	df = pd.DataFrame(filtered_boxes)

	df['ymin'] = df[0].apply(lambda y: max(1,(y*img_height)))
	df['xmin'] = df[1].apply(lambda x: max(1,(x*img_width)))
	df['ymax'] = df[2].apply(lambda y: min(img_height,(y*img_height)))
	df['xmax'] = df[3].apply(lambda x: min(img_width,(x * img_width)))
	boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].to_numpy()

	return boxes_scaled, filtered_classes, filtered_scores, num_det


def draw_boundaryboxes(image_path, annotation_path):

	""" Draw the detection boundary boxes
	args:
		image_path: path to image
		annotation_path: path to groundtruth in Pascal VOC format .xml
	"""
	# Draw detection boundary boxes
	dt_boxes, dt_classes, dt_scores, num_det = postprocess_output(image_path)
	image = cv2.imread(image_path)
	for i in range(num_det):
		if dt_classes[i] == 0:
			[ymin, xmin, ymax, xmax] = list(map(int, dt_boxes[i]))
			cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 1)
			cv2.putText(image, '{}% score'.format(int(dt_scores[i]*100)), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,255,0), 1)

	# Draw groundtruth boundary boxes
	label_map_dict = get_label_map_dict(label_file)
	# Read groundtruth from XML file in Pascal VOC format
	gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)
	for i in range(len(gt_boxes)):
		if gt_classes[i] == 1:
			[ymin, xmin, ymax, xmax] = list(map(int, gt_boxes[i]))
			# cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0, 255, 255), 2)

	saved_path = "out_" + os.path.basename(image_path)
	cv2.imwrite(os.path.join(base_eval_dir, saved_path), image)
	print("Saved at", saved_path)


def voc_parser(path_to_xml_file, label_map_dict):
	"""Parser for Pascal VOC format annotation to TF OD API format
	args:
		path_to_xml_file : path to annotation in Pascal VOC format
		label_map_dict : dictionary of class name to index
	returns
		boxes: array of boundary boxes (m, 4) where each row is [ymin, xmin, ymax, xmax]
		classes: list of class index (m, 1)
		where m is the number of objects
	"""
	boxes = []
	classes = []

	xml = open(path_to_xml_file, "r")
	tree = Et.parse(xml)
	root = tree.getroot()
	xml_size = root.find("size")

	objects = root.findall("object")
	if len(objects) == 0:
		print("No objects for {}")
		return boxes, classes

	obj_index = 0
	for obj in objects:
		class_id = label_map_dict[obj.find("name").text]
		xml_bndbox = obj.find("bndbox")
		xmin = float(xml_bndbox.find("xmin").text)
		ymin = float(xml_bndbox.find("ymin").text)
		xmax = float(xml_bndbox.find("xmax").text)
		ymax = float(xml_bndbox.find("ymax").text)
		boxes.append([ymin, xmin, ymax, xmax])
		classes.append(class_id)
	return boxes, classes


def evaluate_single_image(image_path, annotation_path):
	""" Evaluate mAP on image
	args:
		image_path: path to image
		annotation_path: path to groundtruth in Pascal VOC format .xml
	"""

	categories = create_categories_from_labelmap(label_file)
	label_map_dict = get_label_map_dict(label_file)
	""" 
	 categories,
               include_metrics_per_category=False,
               all_metrics_per_category=False,
               skip_predictions_for_unlabeled_class=False,
               super_categories=None
               """
	coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories=categories, include_metrics_per_category=True)
	image_name = os.path.basename(image_path).split('.')[0]

	# Read groundtruth from XML file in Pascal VOC format
	gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)
	dt_boxes, dt_classes, dt_scores, num_det = postprocess_output(image_path)

	coco_evaluator.add_single_ground_truth_image_info(
		image_id=image_name,
		groundtruth_dict={
			standard_fields.InputDataFields.groundtruth_boxes:
			np.array(gt_boxes),
			standard_fields.InputDataFields.groundtruth_classes:
			np.array(gt_classes)
	})


	coco_evaluator.add_single_detected_image_info(
		image_id=image_name,
		detections_dict={
			standard_fields.DetectionResultFields.detection_boxes:
			dt_boxes,
			standard_fields.DetectionResultFields.detection_scores:
			dt_scores,
			standard_fields.DetectionResultFields.detection_classes:
			dt_classes
		})

	coco_evaluator.evaluate()


def _run_eval():



	image_paths = sorted(glob.glob(data_dir + "/*.png"))
	annotation_paths = sorted(glob.glob(data_dir + "/*.xml"))

	categories = create_categories_from_labelmap(label_file)
	label_map_dict = get_label_map_dict(label_file)
	print(categories)
	coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories=categories, include_metrics_per_category=False)
	counter = 0

	image_list = list()
	groundtruth_annotations_list = list()
	detections_list = list()
	for image_path, annotation_path in zip(image_paths, annotation_paths):
		input_data = prepare_input(image_path)
		interpreter.set_tensor(input_details[0]['index'], input_data)

		start_time = time.time()
		interpreter.invoke()
		stop_time = time.time()
		print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
		boxes, classes, scores, num_det = postprocess_output(image_path)
		draw_boundaryboxes(image_path, annotation_path)
		image_name = os.path.basename(image_path).split('.')[0]

		# Read groundtruth from XML file in Pascal VOC format
		gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)
		dt_boxes, dt_classes, dt_scores, num_det = postprocess_output(image_path)

		coco_evaluator.add_single_ground_truth_image_info(
			image_id=image_name,
			groundtruth_dict={
				standard_fields.InputDataFields.groundtruth_boxes:
					np.array(gt_boxes),
				standard_fields.InputDataFields.groundtruth_classes:
					np.array(gt_classes)
			})
		coco_evaluator.add_single_detected_image_info(
			image_id=image_name,
			detections_dict={
				standard_fields.DetectionResultFields.detection_boxes:
					dt_boxes,
				standard_fields.DetectionResultFields.detection_scores:
					dt_scores,
				standard_fields.DetectionResultFields.detection_classes:
					dt_classes
			})

		# json_pred_output = "pred.json"
		# coco_evaluator.dump_detections_to_json_file(json_pred_output)

		# box_metrics = coco_evaluator.evaluate()
		# print("box_metrics", box_metrics)
		# coco_evaluator.clear()
		# coco_evaluator.clear()

		counter += 1



if __name__ == '__main__':

	_run_eval()
