import os
import cv2
import numpy as np
import random
import argparse
from datetime import datetime

from annotation_util import in_bbox
from ps_dataset import PartScoreDataset

MAX_WINDOW_SIZE = np.asarray([1000, 1800])  # [660, 1300]

DB_ANNOTATIONS = 'annotations'
ANN_TYPE = 'type'
ANN_POINTS = 'points'
ANN_SCORE = 'score'
BBOX = 'bbox'
WIDTH = 'width'
HEIGHT = 'height'


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
        self.contour_img = None
        self.cur_img_name = None
        self.dataset = PartScoreDataset()
        self.cur_annot_type = 'f'
        self.images = []
        self.cur_img_index = 0
        self.scale_xy = np.asarray([1,1])

        self.type_color = {'p': (200, 100, 0),
                           'f': (0, 100, 255),
                           't': (50, 200, 50),
                           'h': (100, 50, 50),
                           'a': (50, 0, 200),
                           'c': (100, 50, 100),
                           'l': (150, 0, 50)}
        self.selected_annot = None
        self.hide_annotations = False

    def load_db(self):
        self.dataset.load(path=self.db_filename)

    def add_annotation(self, img_name, contour, type):
        annot = self.dataset.add_annotation(img_name=img_name, type=type, points=contour)
        self.selected_annot = annot

    def save_backup_db(self):
        orig_db_filename = self.db_filename
        time_str = '_'.join(datetime.now().__str__().split(' ')).split('.')[0]
        self.db_filename = os.path.splitext(self.db_filename)[0] + '.' + time_str + '.json'
        self.save_db()
        self.db_filename = orig_db_filename

    def save_db(self):
        self.dataset.save(path=self.db_filename)

    def window_cords(self, pt):
        return pt * self.scale_xy

    def img_cords(self, pt):
        return pt / self.scale_xy

    def mouse_event(self, event, x, y, flags, param):
        window_pt = np.array([x, y])
        img_pt = self.img_cords(window_pt)

        #print('win: [' + str(x) + ',' + str(y) + '], img: [' + str(img_pt[0]) + ',' + str(img_pt[1]))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_contour = True
            self.refresh_contour_img()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_contour:
                self.cur_contour.append(img_pt.astype(int).tolist())

                # Draw lines while user        is drawing
                if len(self.cur_contour) > 1:
                    pts = np.array(self.cur_contour)
                    cv2.line(self.contour_img,
                             totuple(pts[-2].astype(int)),
                             totuple(pts[-1].astype(int)),
                             color=(10, 55, 255, 100))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_contour = False
            if len(self.cur_contour) > 1:
                self.add_annotation(img_name=self.cur_img_name,
                                    type=self.cur_annot_type,
                                    contour=self.cur_contour
                                    )

                # Clear the current contour
                self.cur_contour = []

            self.refresh_contour_img()
        elif event == cv2.EVENT_RBUTTONUP:
            self.pick(x=img_pt[0], y=img_pt[1])
            self.refresh_contour_img()

    def refresh_contour_img(self):
        self.contour_img = self.cur_img * 0

    def draw_annotations(self, cur_img):
        vis_img = cur_img.copy()

        if self.hide_annotations:
            return vis_img

        # Draw annotation regions
        if self.cur_img_name in self.dataset:
            for ann in self.dataset.get_annotations(img_name=self.cur_img_name):
                pts = np.array(ann[ANN_POINTS]).astype(np.int32)
                cv2.fillPoly(vis_img, [pts], self.type_color[ann[ANN_TYPE]])

        # Draw bounding box around currently selected annot
        if self.selected_annot is not None:
            ann = self.selected_annot

            pt1 = np.array([ann[BBOX]['x'], ann[BBOX]['y']])
            pt2 = np.array([ann[BBOX]['x'] + ann[BBOX][WIDTH], ann[BBOX]['y'] + ann[BBOX][HEIGHT]])

            cv2.rectangle(vis_img,
                          tuple(pt1.astype(int)),
                          tuple(pt2.astype(int)),
                          (0, 0, 0))
            pt_mid = np.array([ann[BBOX]['x'] + ann[BBOX][WIDTH] / 2,
                     ann[BBOX]['y'] + ann[BBOX][HEIGHT] / 2])

            if ANN_SCORE in ann:
                score = ann[ANN_SCORE]
            else:
                score = -1
            cv2.putText(vis_img,
                        str(score),
                        tuple(pt_mid.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        draw_img = (vis_img / 2 + cur_img / 2)
        return draw_img

    def get_rescale_size(self, in_shape, max_size):
        in_shape = np.asarray(in_shape)
        ratio_0 = max_size[0] / float(in_shape[0])
        ratio_1 = max_size[1] / float(in_shape[1])
        ratio = min(ratio_0, ratio_1)
        new_h = int(in_shape[0] * ratio)
        new_w = int(in_shape[1] * ratio)
        return np.array([new_w, new_h])

    def load_image(self, img_name):
        self.selected_annot = None
        print('Loading ' + img_name)
        self.cur_img_name = img_name
        self.cur_img = cv2.imread(os.path.join(self.img_dir, self.cur_img_name))
        if self.cur_img is None:
            print('Could not load image ' + self.cur_img_name)
            return

        img_wh = np.asarray([self.cur_img.shape[1], self.cur_img.shape[0]])

        self.dataset.check_annotations(img_name=self.cur_img_name, img_wh=img_wh)

        self.refresh_contour_img()
        self.draw()

    def next_img(self, find_annot=False, target_image_name=None):
        img_name = None
        for i in range(len(self.images)):
            self.cur_img_index = (self.cur_img_index + 1) % len(self.images)
            img_name = self.images[self.cur_img_index]
            if target_image_name is not None:
                if img_name == target_image_name:
                    break
            elif not find_annot or img_name in self.dataset:
                break

        if img_name is not None:
            if (target_image_name is not None) and (img_name != target_image_name):
                print('Could not find image: ' + target_image_name)

            #print("Loading image: " + img_name)
            self.load_image(img_name)

    def prev_img(self, find_annot=False, target_image_name=None):
        img_name = None
        for i in range(len(self.images)):
            self.cur_img_index = (self.cur_img_index - 1) % len(self.images)
            img_name = self.images[self.cur_img_index]
            if target_image_name is not None:
                if img_name == target_image_name:
                    break
            elif not find_annot or img_name in self.dataset:
                break

        if img_name is not None:
            if (target_image_name is not None) and (img_name != target_image_name):
                print('Could not find image: ' + target_image_name)

            self.load_image(img_name)

    def pick(self, x, y):
        if self.cur_img_name in self.dataset:
            for ann in self.dataset.get_annotations(img_name=self.cur_img_name):
                if in_bbox(ann[BBOX], x, y):
                    self.selected_annot = ann
                    break
        else:
            print('Cant pick, no annotations yet')

    def set_selected_type(self, type_char):
        if self.selected_annot is not None:
            self.selected_annot[ANN_TYPE] = type_char

    def delete_selected_annotation(self):
        if self.cur_img_name in self.dataset and self.selected_annot is not None:
            self.dataset.remove_annotation(img_name=self.cur_img_name, annot=self.selected_annot)
            self.selected_annot = None
            self.refresh_contour_img()

    def draw(self):
        # Compute scaled image size
        scaled_wh = self.get_rescale_size(self.cur_img.shape, self.max_window_size)

        # Compute the scale factor
        orig_wh = np.array([self.cur_img.shape[1], self.cur_img.shape[0]])
        self.scale_xy = scaled_wh.astype(float) / orig_wh

        # Draw annotations on current image
        draw_img = self.draw_annotations(self.cur_img)
        draw_img += self.contour_img

        scaled_draw_img = cv2.resize(draw_img, (scaled_wh[0], scaled_wh[1]))

        # Resize the window
        cv2.resizeWindow(self.cv2_window_name, scaled_wh[0], scaled_wh[1])

        cv2.imshow(self.cv2_window_name, scaled_draw_img)

    def main(self, args):
        self.max_window_size = MAX_WINDOW_SIZE

        # Load annotations
        self.db_filename = args.db_filename
        self.load_db()

        self.save_backup_db()

        # Get list of all files in dir
        self.images = []
        for subdir in self.sub_dirs:
            subdir_path = os.path.join(self.img_dir, subdir)
            print('Adding images in dir ' + subdir_path)
            subdir_files = os.listdir(subdir_path)
            self.images += [os.path.join(subdir, img) for img in subdir_files if
                            img.lower().endswith('.jpg') or img.lower().endswith('.png')]

        if self.images == []:
            print("No images found in " + self.img_dir)
            return

        if args.sort:
            print("Sorting images by mod date...")

            def sort_value(x):
                full_path = os.path.join(self.img_dir, x)
                if os.path.isfile(full_path):
                    return os.path.getmtime(full_path)
                return 0.0

            self.images.sort(key=sort_value, reverse=True)
        else:
            random.shuffle(self.images)

        if args.first_image:
            self.next_img(target_image_name=args.first_image)
        else:
            self.next_img(target_image_name=self.images[0])

        # Create the window
        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.cv2_window_name, self.mouse_event)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            self.draw()
            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                self.hide_annotations = True
                self.refresh_contour_img()
            else:
                self.hide_annotations = False
            if key == ord("s"):
                self.save_db()
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
                if key == ord("1") or key == ord("2") or key == ord("3") or key == ord("4") or key == ord(
                        "5") or key == ord("6") or key == ord("7") or key == ord("8") or key == ord("9") or key == ord(
                        "0"):
                    self.refresh_contour_img()
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
    parser.add_argument('--first_image', '-i', default=None, type=str)
    parser.add_argument('--sort', '-s', action='store_true')
    parser.add_argument('--db_filename', '-j', default=os.path.join('data', 'unscaled_annots.json'))
    parser.add_argument('--image_path', '-p', default=os.path.join('..', 'movies', 'movies'))
    args = parser.parse_args()
    annotator = Annotator(img_dir=args.image_path,
                          sub_dirs=['', os.path.join('..', 't')])
    annotator.main(args)
