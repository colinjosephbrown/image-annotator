import os
import json
import numpy as np
import cv2
import argparse

from annotation_util import get_bbox

BBOX = 'bbox'
WIDTH = 'width'
HEIGHT = 'height'
ANN_TYPE = 'type'
ANN_SCORE = 'score'
ANN_POINTS = 'points'

MAX_SCORE = 10


class Bbox:
    def __init__(self, x, y, w, h):
        self.xy = np.asarray([x, y])
        self.wh = np.asarray([w, h])

    @property
    def x(self):
        return self.xy[0]

    @property
    def y(self):
        return self.xy[1]

    @property
    def width(self):
        return self.wh[0]

    @property
    def height(self):
        return self.wh[1]

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height

    @property
    def xy2(self):
        return np.asarray([self.x2, self.y2])

    @property
    def center(self):
        return self.xy + self.wh / 2


def convert_str(thestr):
    pairs = [('\\', '//'), ('\a', '/a'), ('\b', '/b'), ('\c', '/c'), ('\d', '/d'), ('\e', '/e'), ('\f', '/f'),
             ('\g', '/g'), ('\h', '/h'), ('\i', '/i'), ('\j', '/j'), ('\k', '/k'), ('\l', '/l'), ('\m', '/m'),
             ('\n', '/n'), ('\o', '/o'), ('\p', '/p'), ('\q', '/q'), ('\r', '/r'), ('\s', '/s'), ('\t', '\t'),
             ('\u', '/u'), ('\v', '/v'), ('\w', '/w'), ('\y', '/y'), ('\z', '/z')]
    for p in pairs:
        thestr = thestr.replace(p[0], p[1])
    return thestr


class ImageStats:
    def __init__(self, json_path, img_dir, tile_size=384, num_per_score=5):
        self.img_dir = img_dir
        self.tile_size = tile_size
        self.num_per_score = num_per_score

        self.tiles = []
        self.anns_by_score = []
        self.score_col_index = {}
        self.selected_tile = None

        self.ui_img = None
        self.bg_img = None

        self.annotation_draw_mode = 2
        self.hide_annotations_key_pressed = False

        self.type_color = {'p': (200, 100, 0),
                           'f': (0, 100, 255),
                           't': (50, 200, 50),
                           'h': (100, 50, 50),
                           'a': (50, 0, 200),
                           'c': (100, 50, 100),
                           'l': (150, 0, 50)}

        # Create the window
        self.cv2_window_name = 'chart'
        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.cv2_window_name, self.mouse_event)

        # Load the json db and compute the stats
        self.compute_indexes_and_stats(json_path, recompute_bbox=True)

    def create_chart(self, show_type='p'):
        print("Creating chart...")
        bg_size = (self.tile_size * (MAX_SCORE + 1), self.tile_size * (self.num_per_score + 1))
        self.bg_img = cv2.resize(self.bg_img, bg_size)

        # Create empty grid of tile metadata
        self.tiles = (MAX_SCORE + 1) * [None]
        for score_col in range(len(self.tiles)):
            self.tiles[score_col] = self.num_per_score * [None]

        for score in self.anns_by_score[show_type]:
            self._draw_column(score, show_type)

    def num_annotations(self, ann_type, score):
        return len(self.anns_by_score[ann_type][score])

    def _draw_column(self, score, type):
        if score not in self.score_col_index:
            self.score_col_index[score] = 0

        num_anns = self.num_annotations(ann_type=type, score=score)
        max_num_rows = min(self.num_per_score, num_anns)

        # Fill score column
        row_index = 0
        loop_count = 0
        while row_index < max_num_rows:
            cur_index = self.score_col_index[score]
            try:
                (img_name, ann) = self.anns_by_score[type][score][cur_index]
                img_name = convert_str(img_name)
            except IndexError as e:
                print('cur_index: ' + str(cur_index))
                print('score:' + str(score))
                print('type:' + str(type))
                print(e)
                break #TODO: This is causing problems... figure it out

            if self.score_col_index[score] + 1 == num_anns:
                loop_count += 1

            if loop_count == 2:
                break

            self.score_col_index[score] = (self.score_col_index[score] + 1) % num_anns

            if not self._draw_tile(ann, score, img_name, row_index):
                continue

            # Increment index
            row_index += 1

    def _draw_tile(self, ann, score, img_name, row_index):
        img_path = os.path.join(self.img_dir, img_name)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print("Image " + img_name + " cannot be loaded...")
            return False

        imh, imw = raw_img.shape[:2]

        # Make bbox square
        if ann[BBOX][WIDTH] > ann[BBOX][HEIGHT]:
            ann[BBOX]['y'] -= (ann[BBOX][WIDTH] - ann[BBOX][HEIGHT]) / 2
            ann[BBOX][HEIGHT] = ann[BBOX][WIDTH]
        else:
            ann[BBOX]['x'] -= (ann[BBOX][HEIGHT] - ann[BBOX][WIDTH]) / 2
            ann[BBOX][WIDTH] = ann[BBOX][HEIGHT]

        # Draw segmentation
        pts = np.asarray(ann[ANN_POINTS])
        cv2.polylines(raw_img, [pts], False, self.type_color[ann[ANN_TYPE]], thickness=4)

        # Crop to ann bbox
        x1 = np.clip(ann[BBOX]['x'], 0, imw).astype(int)
        x2 = np.clip(ann[BBOX]['x'] + ann[BBOX][WIDTH], 0, imw).astype(int)
        y1 = np.clip(ann[BBOX]['y'], 0, imh).astype(int)
        y2 = np.clip(ann[BBOX]['y'] + ann[BBOX][HEIGHT], 0, imh).astype(int)
        if x2 <= x1 or y2 <= y1:
            print("Annotation has negative dim in image" + img_name)
            return False
        try:
            cropped_img = raw_img[y1:y2, x1:x2]
        except TypeError as e:
            print(str(x1) + ' ' + str(x2) + ' ' + str(y1) + ' ' + str(y2))
            raise

        # Resize tile
        cropped_img = cv2.resize(cropped_img, (self.tile_size, self.tile_size))

        # Draw at x, y
        tile_bbox = self.get_tile_bbox(score, row_index)
        self.bg_img[tile_bbox.y:tile_bbox.y2, tile_bbox.x:tile_bbox.x2] = cropped_img

        # Set tile
        self.tiles[score][row_index] = {'file_name': img_name, 'annotation': ann}
        return True

    def get_tile_bbox(self, score, row_index):
        # TODO: add offset and row/col padding
        bbox = Bbox(x=score * self.tile_size,
                    y=row_index * self.tile_size,
                    w=self.tile_size,
                    h=self.tile_size)
        return bbox

    def get_tile_col_row(self, x, y):
        # TODO: add offset and row/col padding
        tile_col = int(x / self.tile_size)
        tile_row = int(y / self.tile_size)
        return tile_col, tile_row

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            score, row_index = self.get_tile_col_row(x, y)
            self.selected_tile = (score, row_index)
            tile = self.tiles[score][row_index]
            if tile is not None:
                print('Tile[' + str(score) + ', ' + str(row_index) + ']: ' + tile['file_name'])
        elif event == cv2.EVENT_RBUTTONUP:
            self.selected_tile = None
            pass

    #def set_selected_type(self, type):
    #    self.create_chart(bg_img, show_type=type)

    def compute_indexes_and_stats(self, json_fn, recompute_bbox=False):
        data = json.load(open(json_fn, 'r'))
        counts = {}
        total_count = 0
        self.anns_by_score = {}
        for img_name in data:
            for ann in data[img_name]['annotations']:
                total_count += 1
                if ann[ANN_TYPE] not in counts:
                    counts[ann[ANN_TYPE]] = {}
                if ann[ANN_SCORE] not in counts[ann[ANN_TYPE]]:
                    counts[ann[ANN_TYPE]][ann[ANN_SCORE]] = 0
                counts[ann[ANN_TYPE]][ann[ANN_SCORE]] += 1

                if ann[ANN_TYPE] not in self.anns_by_score:
                    self.anns_by_score[ann[ANN_TYPE]] = {}
                if ann[ANN_SCORE] not in self.anns_by_score[ann[ANN_TYPE]]:
                    self.anns_by_score[ann[ANN_TYPE]][ann[ANN_SCORE]] = []
                self.anns_by_score[ann[ANN_TYPE]][ann[ANN_SCORE]].append((img_name, ann))

                if recompute_bbox:
                    #print('Recomputing bbox for annot in img ' + img_name)
                    ann[BBOX] = get_bbox(np.array(ann['points']))

        # Output counts
        print(str(total_count) + " annotations in total, across " + str(len(data)) + " annotated images.")
        print("Score summary:")
        for type in counts:
            output = str(type) + '| '
            for i in range(0, max(counts[type].keys()) + 1):
                if i not in counts[type]:
                    counts[type][i] = 0
                # output += str(i) + ':' + str(counts[type][i]) + ', '
                output += str(counts[type][i]) + ', '
            print(output)

    def main(self):
        self.bg_img = cv2.imread('black.jpg')

        while True:
            new_annot_type = None
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            annot_type_keys = ['a', 'c', 'f', 'h', 'l', 'p', 't']

            for k in annot_type_keys:
                if key == ord(k):
                    new_annot_type = k

                if key == 32:
                    self.hide_annotations_key_pressed = True
                else:
                    if self.hide_annotations_key_pressed:
                        self.annotation_draw_mode += 1
                        if self.annotation_draw_mode > 2:
                            self.annotation_draw_mode = 0
                    self.hide_annotations_key_pressed = False

            if new_annot_type is not None:
                self.create_chart(show_type=new_annot_type)

            # UI
            draw_img = self.bg_img.copy()

            if self.selected_tile is not None:
                score = self.selected_tile[0]
                row_index = self.selected_tile[1]
                tile = self.tiles[score][row_index]

                if tile is not None:
                    tile_bbox = self.get_tile_bbox(score, row_index)
                    cv2.rectangle(draw_img, pt1=tuple(tile_bbox.xy), pt2=tuple(tile_bbox.xy2), color=(255, 255, 0), thickness=3)

                    cv2.putText(draw_img,
                                str(score),
                                tuple(tile_bbox.center.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow(self.cv2_window_name, draw_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", type=str)
    args = parser.parse_args()
    json_path = os.path.join('data', 'unscaled_annots.json')
    stats = ImageStats(json_path=json_path,
                       tile_size=384,
                       num_per_score=5,
                       img_dir=os.path.join('..', 'movies', 'movies'))
    stats.main()
