import json
import numpy as np

from annotation_util import get_bbox

FILE_NAME = 'file_name'
POINTS = 'points'
TYPE = 'type'
SCORE = 'score'
BBOX = 'bbox'
ANNOTATIONS = 'annotations'
WIDTH = 'width'
HEIGHT = 'height'


class PartScoreAnnotation(object):
    def __init__(self, obj):
        self.obj = obj

    @property
    def points(self):
        return self.obj[POINTS]

    @points.setter
    def points(self, value):
        self.obj[POINTS] = value

    @property
    def type(self):
        return self.obj[TYPE]

    @type.setter
    def type(self, value):
        self.obj[TYPE] = value

    @property
    def score(self):
        return self.obj[SCORE]

    @score.setter
    def score(self, value):
        self.obj[SCORE] = value

    @property
    def bbox(self):
        return self.obj[BBOX]

    @bbox.setter
    def bbox(self, value):
        self.obj[BBOX] = value


class PartScoreDataset(object):
    def __init__(self, path=None):
        self.data = {}

        if path is not None:
            self.load(path)

    def __getitem__(self, img_name):
        return self.get_image_annotation(img_name)

    def __contains__(self, img_name):
        return img_name in self.data

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)

    def load(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

        num_anns = 0
        for img_name, img_ann in self.data.iteritems():
            num_anns += len(img_ann[ANNOTATIONS])

        print("Found " + str(num_anns) + ' annotations in ' + str(len(self.data))
              + " annotated images in " + path)

    def save(self, path):
        print('Saving...')
        with open(path, 'w') as f:
            json.dump(path, f)
        print('Saved to ' + path)

    def get_image_annotation(self, img_name):
        return self.data[img_name]

    def get_annotations(self, img_name):
        return self.data[img_name][ANNOTATIONS]

    def add_annotation(self, img_name, type, points, score=0, bbox=None):
        # Add image annotation if it doesn't exist
        if img_name not in self.data:
            img_annot = {FILE_NAME: img_name, ANNOTATIONS: []}
            self.data[img_name] = img_annot

        # Add annotation
        if bbox is None:
            bbox = get_bbox(np.array(points, np.int32))

        annot = {TYPE: type,
                 POINTS: points,
                 SCORE: score,
                 BBOX: bbox}

        self.data[img_name][ANNOTATIONS].append(annot)

        return annot

    def remove_annotation(self, img_name, annot):
        self.get_annotations(img_name=img_name).remove(annot)

    def check_annotations(self, img_name, img_wh):
        if img_name not in self.data:
            return

        remove_list = []
        for ann in self.get_annotations(img_name=img_name):
            ann_valid = True

            if ann[BBOX]['x'] + ann[BBOX][WIDTH] <= 0:
                ann_valid = False
            elif ann[BBOX]['x'] > img_wh[0]:
                ann_valid = False
            elif ann[BBOX]['y'] + ann[BBOX][HEIGHT] <= 0:
                ann_valid = False
            elif ann[BBOX]['y'] > img_wh[1]:
                ann_valid = False

            if not ann_valid:
                remove_list.append(ann)

        if len(remove_list) > 0:
            print(remove_list)

        for ann in remove_list:
            print('Deleting annotation out of frame...')
            self.get_annotations(img_name=img_name).remove(ann)



