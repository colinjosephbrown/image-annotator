import os
import cv2
import numpy as np
import json
import copy
import random
import argparse

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class Annotator:

	def __init__(self, img_dir, sub_dirs=None):
		self.img_dir = img_dir
		self.sub_dirs = [] if sub_dirs is None else sub_dirs
		self.cv2_window_name = "annotator"
		self.cur_contour = []
		self.drawing_contour = False
		self.cur_img = None
		self.vis_img = None
		self.cur_img_name = None
		self.annots_dic = {}
		self.cur_annot_type = 'f'
		self.images = []
		self.cur_img_index = 0
		
		self.type_color = {'p': (200, 100, 0), 
							'f': (0, 100, 255), 
							't': (50, 200, 50), 
							'h': (100, 50, 50), 
							'a': (50, 0, 200), 
							'c': (100, 50, 100), 
							'l': (150, 0, 50)}
		self.selected_annot = None
		self.hide_annotations = False
	
	def window_cords(self, pt):
		return pt * self.scale_factor
	
	def img_cords(self, pt):
		return pt / self.scale_factor

	def mouse_event(self, event, x, y, flags, param):
		window_pt = np.array([x, y])
		img_pt = self.img_cords(window_pt)
		#print(str(img_pt) + '=' + str(window_pt) + '/' + str(self.scale_factor))
		
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing_contour = True
			self.refresh_vis_img()
		elif event == cv2.EVENT_MOUSEMOVE:
			if self.drawing_contour:
				self.cur_contour.append(img_pt.astype(int).tolist())
				
				# Draw lines while user is drawing
				if len(self.cur_contour) > 1: 
					pts = self.window_cords(np.array(self.cur_contour))
					cv2.line(self.vis_img, 
							 totuple(pts[-2].astype(int)), 
							 totuple(pts[-1].astype(int)), 
							 color=(10,55,255,100))
		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing_contour = False
			if len(self.cur_contour) > 1: 
				self.add_annotation()
			self.refresh_vis_img()
		elif event == cv2.EVENT_RBUTTONUP:
			self.pick(x=img_pt[0], y=img_pt[1])
			self.refresh_vis_img()
			
	def refresh_vis_img(self):
		self.vis_img = self.cur_img.copy()
		self.draw_img = self.cur_img.copy()
	
	def draw_annotations(self):
		if self.hide_annotations:
			return
		
		# Draw annotation regions
		if self.cur_img_name in self.annots_dic:
			for ann in self.annots_dic[self.cur_img_name]['annotations']:
				pts = self.window_cords(np.array(ann['points'])).astype(np.int32)
				cv2.fillPoly( self.vis_img, [pts], self.type_color[ann['type']])
		
		# Draw bounding box around currently selected annot
		if self.selected_annot is not None:
			ann = self.selected_annot
			
			pt1 = self.window_cords(np.array([ann['bbox']['x'], 
												ann['bbox']['y']]))
			pt2 = self.window_cords(np.array([ann['bbox']['x'] + ann['bbox']['width'], 
												ann['bbox']['y'] + ann['bbox']['height']]))
			
			cv2.rectangle(self.vis_img, 
							tuple(pt1.astype(int)), 
							tuple(pt2.astype(int)), 
							(0,0,0))
			pt_mid = self.window_cords(np.array([ann['bbox']['x'] + ann['bbox']['width']/2, 
												ann['bbox']['y'] + ann['bbox']['height']/2]))
			cv2.putText(self.vis_img, 
						str(ann['score']), 
						tuple(pt_mid.astype(int)), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		
		self.draw_img = (self.vis_img / 2 + self.cur_img / 2) 
	
	def add_annotation(self):
		# Add image annotation if it doesn't exist
		if self.cur_img_name not in self.annots_dic:
			img_annot = {'file_name': self.cur_img_name, 'annotations': []}
			self.annots_dic[self.cur_img_name] = img_annot
		
		# Add annotation
		bbox = self.get_bbox(np.array(self.cur_contour, np.int32))
		annot = {'type': self.cur_annot_type, 
				'points': self.cur_contour, 
				'score': 0, 
				'bbox': bbox}
		self.selected_annot = annot
		self.annots_dic[self.cur_img_name]['annotations'].append(annot)
		
		# Clear the current contour
		self.cur_contour = []
	
	def save_annotations(self):
		json.dump(self.annots_dic, open('unscaled_annots.json', 'w'))
	
	def resize_img(self, cur_img, max_size):
		ratio_0 = max_size[0] / float(cur_img.shape[0])
		ratio_1 = max_size[1] / float(cur_img.shape[1])
		ratio = min(ratio_0, ratio_1)
		new_h = int(cur_img.shape[0] * ratio)
		new_w = int(cur_img.shape[1] * ratio)
		cur_img = cv2.resize(cur_img, (new_w, new_h))
		return cur_img
		
	def load_image(self, img_name):
		self.selected_annot = None
		print(img_name)
		self.cur_img_name = img_name
		self.cur_img = cv2.imread(os.path.join(self.img_dir, self.cur_img_name))
		if self.cur_img is None:
			print('Could not load image ' + self.cur_img_name)
			return
		
		# Resize the image
		imh, imw = self.cur_img.shape[:2]
		orig_size = np.array([imw, imh])
		self.cur_img = self.resize_img(self.cur_img, self.max_window_size)
		new_h, new_w = self.cur_img.shape[:2]
		scaled_size = np.array([new_w, new_h])
		
		# Compute the scale factor
		self.scale_factor = scaled_size.astype(float) / orig_size
		
		# Resize the window
		cv2.resizeWindow(self.cv2_window_name, new_w, new_h)
		
		# Copy the new image to the vis and draw buffers
		self.vis_img = self.cur_img.copy()
		self.draw_img = self.cur_img.copy()
	
	def next_img(self, find_annot=False, target_image_name=None):
		for i in range(len(self.images)):
			self.cur_img_index = (self.cur_img_index + 1) % len(self.images)
			img_name = self.images[self.cur_img_index]
			if target_image_name is not None:
				if img_name == target_image_name:
					print("Loading target image: " + target_image_name)
					break
			elif not find_annot or img_name in self.annots_dic:
				break
		self.load_image(img_name)
	
	def prev_img(self, find_annot=False, target_image_name=None):
		for i in range(len(self.images)):
			self.cur_img_index = (self.cur_img_index - 1) % len(self.images)
			img_name = self.images[self.cur_img_index]
			if target_image_name is not None:
				if img_name == target_image_name:
					break
			elif not find_annot or img_name in self.annots_dic:
				break
		self.load_image(img_name)
	
	def get_bbox(self, pts):
		bbox = {'x': pts[:,0].min(), 'y': pts[:,1].min()}
		bbox['width'] = pts[:,0].max() - bbox['x']
		bbox['height'] = pts[:,1].max() - bbox['y']
		return bbox
	
	def pick(self, x, y):
		if self.cur_img_name in self.annots_dic:
			for ann in self.annots_dic[self.cur_img_name]['annotations']:
				if self.in_bbox(ann['bbox'], x, y):
					self.selected_annot = ann
					break
		else:
			print('Cant pick, no annotations yet')
	
	def in_bbox(self, bbox, x, y):
		return x >= bbox['x'] and x <= bbox['x'] + bbox['width'] and y >= bbox['y'] and y <= bbox['y'] + bbox['height']
		
	def set_selected_type(self, type_char):
		if self.selected_annot is not None:
			self.selected_annot['type'] = type_char
	
	def delete_selected_annotation(self):
		if self.cur_img_name in self.annots_dic and self.selected_annot is not None:
			self.annots_dic[self.cur_img_name]['annotations'].remove(self.selected_annot)
			self.selected_annot = None
			self.refresh_vis_img()
	
	def main(self, args):
		self.max_window_size = [660, 1300]
		
		#Load annotations
		self.annots_dic = json.load(open('unscaled_annots.json', 'r'))
		print("Found " + str(len(self.annots_dic)) + " images with annotations.")
		
		# Get list of all files in dir
		self.images = []
		for subdir in self.sub_dirs:
			subdir_path = os.path.join(self.img_dir, subdir)
			print(subdir_path)
			subdir_files = os.listdir(subdir_path)
			self.images += [ os.path.join(subdir, img) for img in subdir_files if img.lower().endswith('.jpg') or img.lower().endswith('.png')]
		
		if self.images == []:
			print "No images found in " + self.img_dir
			return
		
		if args.sort:
			print("Sorting images by mod date...")
			def sort_value(x):
				full_path = os.path.join(self.img_dir,x)
				if os.path.isfile(full_path):
					return os.path.getmtime(full_path)
				return 0.0
			self.images.sort(key=sort_value, reverse=True)
			self.next_img(target_image_name=self.images[0])
		else:
			random.shuffle(self.images)
			self.next_img(target_image_name=args.first_image)
		
		# Create the window
		cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)
		cv2.setMouseCallback(self.cv2_window_name, self.mouse_event)
		 
		# keep looping until the 'q' key is pressed
		while True:
			# display the image and wait for a keypress
			cv2.imshow(self.cv2_window_name, self.draw_img)
			self.draw_annotations()
			key = cv2.waitKey(1) & 0xFF
			
			if key == 32:
				self.hide_annotations = True
				self.refresh_vis_img()
			else:
				self.hide_annotations = False
			if key == ord("s"):
				self.save_annotations()
			elif key == ord("."):
				self.next_img()
			elif key == ord(","):
				self.prev_img()
			elif key == ord(">"):
				self.next_img(find_annot=True)
			elif key == ord("<"):
				self.prev_img(find_annot=True)
			elif key == ord("d"):
				self.delete_selected_annotation()
			elif key == ord("p"):
				self.cur_annot_type = 'p'
				self.set_selected_type(self.cur_annot_type)
			elif key == ord("f"):
				self.cur_annot_type = 'f'
				self.set_selected_type(self.cur_annot_type)
			elif key == ord("t"):
				self.cur_annot_type = 't'
				self.set_selected_type(self.cur_annot_type)
			elif key == ord("h"):
				self.cur_annot_type = 'h'
				self.set_selected_type(self.cur_annot_type)
			elif key == ord("a"):
				self.cur_annot_type = 'a'
				self.set_selected_type(self.cur_annot_type)
			elif key == ord("c"):
				self.cur_annot_type = 'c'
				self.set_selected_type(self.cur_annot_type)
			elif key == ord("l"):
				self.cur_annot_type = 'l'
				self.set_selected_type(self.cur_annot_type)
			elif key == 27:
				break
			elif self.selected_annot is not None:
				if key == ord("1") or key == ord("2") or key == ord("3") or key == ord("4") or key == ord("5") or key == ord("6") or key == ord("7") or key == ord("8") or key == ord("9") or key == ord("0"):
					self.refresh_vis_img()
				if key == ord("1"):
					self.selected_annot['score'] = 1
				elif key == ord("2"):
					self.selected_annot['score'] = 2
				elif key == ord("3"):
					self.selected_annot['score'] = 3
				elif key == ord("4"):
					self.selected_annot['score'] = 4
				elif key == ord("5"):
					self.selected_annot['score'] = 5
				elif key == ord("6"):
					self.selected_annot['score'] = 6
				elif key == ord("7"):
					self.selected_annot['score'] = 7
				elif key == ord("8"):
					self.selected_annot['score'] = 8
				elif key == ord("9"):
					self.selected_annot['score'] = 9
				elif key == ord("0"):
					self.selected_annot['score'] = 10
		
		# close all open windows
		cv2.destroyAllWindows()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--first_image", "-i", default=None, type=str)
	parser.add_argument("--sort", "-s", action="store_true")
	args = parser.parse_args()
	annotator = Annotator(img_dir=os.path.join('movies', 'movies'), 
						  sub_dirs=['', os.path.join('..', 't')])
	annotator.main(args)
