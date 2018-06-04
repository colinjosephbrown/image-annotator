import os
import json
import numpy as np
import cv2
import argparse

def resize_img(cur_img, max_size=(660, 1300)):
	ratio_0 = max_size[0] / float(cur_img.shape[0])
	ratio_1 = max_size[1] / float(cur_img.shape[1])
	ratio = min(ratio_0, ratio_1)
	new_h = int(cur_img.shape[0] * ratio)
	new_w = int(cur_img.shape[1] * ratio)
	cur_img = cv2.resize(cur_img, (new_w, new_h))
	return cur_img

def convert_str(thestr):
	pairs = [('\\', '//'), ('\a','/a'), ('\b','/b'), ('\c','/c'), ('\d','/d'), ('\e','/e'),('\f','/f'), ('\g','/g'), ('\h','/h'), ('\i','/i'), ('\j', '/j'), ('\k', '/k'), ('\l', '/l'), ('\m', '/m'), ('\n', '/n'), ('\o', '/o'), ('\p', '/p'), ('\q', '/q'), ('\r', '/r'), ('\s', '/s'), ('\t','\t'), ('\u', '/u'), ('\v', '/v'), ('\w', '/w'), ('\y', '/y'), ('\z', '/z')]
	for p in pairs:
		thestr = thestr.replace(p[0], p[1])
	return thestr

class ImageStats:
	def __init__(self):
		self.tiles = []
		self.anns_by_score = []
		self.score_col_index = {}
		
	def create_chart(self, bg_img, w, n, img_dir, show_type='p'):
		print("Creating chart...")
		bg_size = (w*11, w*(n+1))
		bg_img = cv2.resize(bg_img, bg_size)
		
		# Create empty grid of tile metadata
		self.tiles = 11*[None]
		for score_col in range(len(self.tiles)):
			self.tiles[score_col] = n*[None]
		
		type = show_type

		for score in self.anns_by_score[type]:
			if score not in self.score_col_index:
				self.score_col_index[score] = 0
			num_anns = len(self.anns_by_score[type][score])
			max_i = min(n, num_anns)
			
			# Fill score column
			i = 0
			loop_count = 0
			while i < max_i:
				cur_index = self.score_col_index[score]
				try:
					(img_name, ann) = self.anns_by_score[type][score][cur_index]
				except IndexError as e:
					print(e)
				if self.score_col_index[score] + 1 == num_anns:
					loop_count += 1
				if loop_count == 2:
					break
					
				self.score_col_index[score] = (self.score_col_index[score] + 1) % num_anns
				
				
				# Load image
				img_name = convert_str(img_name)
				print('New safe img_name')
				raw_img = cv2.imread( os.path.join(img_dir, img_name))
				if raw_img is None:
					print("Image " + img_name + " cannot be loaded...")
					continue
				raw_img = resize_img(raw_img)
				imh, imw = raw_img.shape[:2]
				
				# Make bbox square
				if ann['bbox']['width'] > ann['bbox']['height']:
					ann['bbox']['y'] -= (ann['bbox']['width'] - ann['bbox']['height']) / 2
					ann['bbox']['height'] = ann['bbox']['width']
				else:
					ann['bbox']['x'] -= (ann['bbox']['height'] - ann['bbox']['width']) / 2
					ann['bbox']['width'] = ann['bbox']['height']
				
				# Crop to ann bbox
				x1 = np.clip(ann['bbox']['x'], 0, imw).astype(int)
				x2 = np.clip(ann['bbox']['x']+ ann['bbox']['width'], 0, imw).astype(int)
				y1 = np.clip(ann['bbox']['y'], 0, imh).astype(int)
				y2 = np.clip(ann['bbox']['y']+ ann['bbox']['height'], 0, imh).astype(int)
				if x2 <= x1 or y2 <= y1:
					print("Annotation has negative dim in image" + img_name )
					continue
				try:
					cropped_img = raw_img[y1:y2, x1:x2]
				except TypeError as e:
					print(str(x1) + ' ' + str(x2) + ' ' + str(y1) + ' ' + str(y2))
					raise
				
				# Resize to 100x100
				cropped_img = cv2.resize(cropped_img, (w, w))
				
				# Draw at x, y
				bg_img[i*w:(i+1)*w, score*w:(score+1)*w] = cropped_img
				
				# Set tile
				self.tiles[score][i] = {'file_name':img_name, 'annotation':ann}
				
				# Increment index
				i += 1
					
		cv2.imshow(self.cv2_window_name, bg_img)

	def mouse_event(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			pass
		elif event == cv2.EVENT_MOUSEMOVE:
			pass
		elif event == cv2.EVENT_LBUTTONUP:
			#TODO: Display name of image in current tile
			tile_x = int(x / self.tile_size)
			tile_y = int(y / self.tile_size)
			print(self.tiles[tile_x][tile_y]['file_name'])
			pass
		elif event == cv2.EVENT_RBUTTONUP:
			pass
	
	def set_selected_type(self, type):
		self.create_chart(bg_img, w, n, img_dir, show_type=type)
	
	def compute_indexes_and_stats(self, json_fn):
		data = json.load(open(json_fn, 'r'))
		counts = {}
		total_count = 0
		self.anns_by_score = {}
		for img_name in data:
			for ann in data[img_name]['annotations']:
				total_count += 1
				if ann['type'] not in counts:
					counts[ann['type']] = {}
				if ann['score'] not in counts[ann['type']]:
					counts[ann['type']][ann['score']] = 0 
				counts[ann['type']][ann['score']] += 1
				
				if ann['type'] not in self.anns_by_score:
					self.anns_by_score[ann['type']] = {}
				if ann['score'] not in self.anns_by_score[ann['type']]:
					self.anns_by_score[ann['type']][ann['score']] = []
				self.anns_by_score[ann['type']][ann['score']].append( (img_name, ann) )
		
		# Output counts
		print(str(total_count) + " annotations in total, across " + str(len(data)) + " annotated images.")
		print("Score summary:")
		for type in counts:
			output = str(type) + '| '
			for i in range(0, max(counts[type].keys())+1):
				if i not in counts[type]:
					counts[type][i] = 0
				#output += str(i) + ':' + str(counts[type][i]) + ', '
				output += str(counts[type][i]) + ', '
			print(output)
		
	def main(self, args):
		# Create the window
		self.cv2_window_name = 'chart'
		cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)
		cv2.setMouseCallback(self.cv2_window_name, self.mouse_event)
		
		# Load the json db and compute the stats
		fn = 'annotations.json'
		self.compute_indexes_and_stats(fn)
		
		# Create sample chart
		img_dir = os.path.join('movies', 'movies')
		self.tile_size = 100
		n = 5
		bg_img = cv2.imread('black.jpg')

		while (True):			
			key = cv2.waitKey(1) & 0xFF
			
			if key == ord("p"):
				self.cur_annot_type = 'p'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == ord("f"):
				self.cur_annot_type = 'f'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == ord("t"):
				self.cur_annot_type = 't'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == ord("h"):
				self.cur_annot_type = 'h'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == ord("a"):
				self.cur_annot_type = 'a'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == ord("c"):
				self.cur_annot_type = 'c'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == ord("l"):
				self.cur_annot_type = 'l'
				self.create_chart(bg_img, self.tile_size, n, img_dir, show_type=self.cur_annot_type)
			elif key == 27:
				break

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--type", "-t", type=str)
	args = parser.parse_args()
	stats = ImageStats()
	stats.main(args)
		
	
	
	
