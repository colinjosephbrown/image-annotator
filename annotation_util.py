WIDTH = 'width'
HEIGHT = 'height'

def get_bbox(pts):
	bbox = {'x': int(pts[:, 0].min()), 'y': int(pts[:, 1].min())}
	bbox['width'] = int(pts[:, 0].max() - bbox['x'])
	bbox['height'] = int(pts[:, 1].max() - bbox['y'])
	return bbox

def in_bbox(bbox, x, y):
	return x >= bbox['x'] and x <= bbox['x'] + bbox[WIDTH] and y >= bbox['y'] and y <= bbox['y'] + bbox[HEIGHT]