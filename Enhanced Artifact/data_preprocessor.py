import random
from typing import Tuple
from typing import Optional, Tuple
import cv2
import numpy as np

from data_loader import Batch


class Preprocessor:
    def __init__(self,
                 img_size: Tuple[int, int],
                 padding: int = 0,
                 dynamic_width: bool = False,
                 data_augmentation: bool = False,
                 line_mode: bool = False,
                 lmdb_path: Optional[str] = None) -> None:
        # dynamic width only supported when no data augmentation happens
        assert not (dynamic_width and data_augmentation)
        # when padding is on, we need dynamic width enabled
        assert not (padding > 0 and not dynamic_width)

        self.img_size = img_size
        self.padding = padding
        self.dynamic_width = dynamic_width
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode
        self.lmdb_path = lmdb_path

    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            "check if the char is a duplicate then +2 to the cost"
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                "+1 if it a special char"
                cost += 1
                "the cost need to be less than the max length of the word"
            if cost > max_text_len:
                return text[:i]
        return text

    '''
    # This function is use to process line of words - will not be necessary for the project     
    def _simulate_text_line(self, batch: Batch) -> Batch:
        """Create image of a text line by pasting multiple word images into an image."""

        default_word_sep = 30
        default_num_words = 5

        # go over all batch elements
        res_imgs = []
        res_gt_texts = []
        for i in range(batch.batch_size):
            # number of words to put into current line
            num_words = random.randint(1, 8) if self.data_augmentation else default_num_words

            # concat ground truth texts
            curr_gt = ' '.join([batch.gt_texts[(i + j) % batch.batch_size] for j in range(num_words)])
            res_gt_texts.append(curr_gt)

            # put selected word images into list, compute target image size
            sel_imgs = []
            word_seps = [0]
            h = 0
            w = 0
            for j in range(num_words):
                curr_sel_img = batch.imgs[(i + j) % batch.batch_size]
                curr_word_sep = random.randint(20, 50) if self.data_augmentation else default_word_sep
                h = max(h, curr_sel_img.shape[0])
                w += curr_sel_img.shape[1]
                sel_imgs.append(curr_sel_img)
                if j + 1 < num_words:
                    w += curr_word_sep
                    word_seps.append(curr_word_sep)

            # put all selected word images into target image
            target = np.ones([h, w], np.uint8) * 255
            x = 0
            for curr_sel_img, curr_word_sep in zip(sel_imgs, word_seps):
                x += curr_word_sep
                y = (h - curr_sel_img.shape[0]) // 2
                target[y:y + curr_sel_img.shape[0]:, x:x + curr_sel_img.shape[1]] = curr_sel_img
                x += curr_sel_img.shape[1]

            # put image of line into result
            res_imgs.append(target)

        return Batch(res_imgs, res_gt_texts, batch.batch_size)'''

    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size, apply data augmentation."""

        # there are damaged files in IAM dataset - just use black image instead
        if img is None:
            '''Add a message to locate the error'''
            print("Warning: Image is None, empty, or not loaded correctly. Using a black placeholder image.")
            img = np.zeros(self.img_size[::-1])

        # image augmentation
        '''converting image to float to agument'''
        img = img.astype(np.float64)
        if self.data_augmentation:
            # photometric data augmentation 
            ''' each image in the pool have 25% chance to get GaussianBlur'''
            if random.random() < 0.25:
                def rand_odd():
                    '''generate a random number either 3 5 or 7 '''
                    return random.randint(1, 3) * 2 + 1
                ''' deteermine intensity of blur base on the rand_odd '''
                img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
            if random.random() < 0.25:
                '''thickening and expand the details '''
                img = cv2.dilate(img, np.ones((3, 3)))
            if random.random() < 0.25:
                ''' erode or shrink the details'''
                img = cv2.erode(img, np.ones((3, 3)))

            # geometric data augmentation
            ''' declare variable to contain the target image sizes'''
            wt, ht = self.img_size
            ''' the current image size '''
            h, w = img.shape
            ''' scaling function and pick the smallest value'''
            f = min(wt / w, ht / h)
            ''' generate random floating points 0.75 and 1.05 to determine the x and y'''
            fx = f * np.random.uniform(0.75, 1.05)
            fy = f * np.random.uniform(0.75, 1.05)

            # random position around center 
            '''using the size of the scaling factor to find the image center '''
            txc = (wt - w * fx) / 2
            tyc = (ht - h * fy) / 2
            ''' calculate the space that the image can shift base as the function pick the max value between the calculated and 0'''
            freedom_x = max((wt - fx * w) / 2, 0)
            freedom_y = max((ht - fy * h) / 2, 0)
            ''' translations image : calcuate the x and y to move the image'''
            tx = txc + np.random.uniform(-freedom_x, freedom_x)
            ty = tyc + np.random.uniform(-freedom_y, freedom_y)

            # map image into target image
            M = np.float32([[fx, 0, tx], [0, fy, ty]])
            ''' reverse the image andd switching the target sizes - see Affine transformation'''
            target = np.ones(self.img_size[::-1]) * 255
            ''' apply the changes to the image -- see Affine transformation'''
            img = cv2.warpAffine(img, M, dsize=self.img_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

            # photometric data augmentation
            '''' each random image have 50% chance'''
            if random.random() < 0.5:
                ''' adding pixel value to each pixel make it brighter'''
                img = img * (0.25 + random.random() * 0.75)
            if random.random() < 0.25:
                ''' adding noises to the image'''
                img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 25), 0, 255)
            if random.random() < 0.1:
                ''' inverting image color, white to black and vice versa'''
                img = 255 - img

        # no data augmentation needs 
        else:
            ''' adjust the width size of the image'''
            if self.dynamic_width:
                ht = self.img_size[1]
                h, w = img.shape
                f = ht / h
                wt = int(f * w + self.padding)
                wt = wt + (4 - wt) % 4
                tx = (wt - w * f) / 2
                ty = 0
            else:
                ''' adjust the height size of the image'''
                wt, ht = self.img_size
                h, w = img.shape
                f = min(wt / w, ht / h)
                tx = (wt - w * f) / 2
                ty = (ht - h * f) / 2

            # map image into target image
            ''' function ensure the image is entered'''
            M = np.float32([[f, 0, tx], [0, f, ty]])
            ''' function ensure the image is ideal size'''
            target = np.ones([ht, wt]) * 255
            ''' use affine transformation to make image in a proper angle'''
            img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

        # transpose for TF
        img = cv2.transpose(img)

        # convert to range [-1, 1]
        ''' normalization the image by divide all pixel tooo maximum amount of pixel to get the range between 0-1 then move the graph to -0.5'''
        img = img / 255 - 0.5
        return img

    def process_batch(self, batch: Batch) -> Batch:
        ''' # this line check if the line_mode (mutiple words present) is true which is not need for the project
        if self.line_mode:
            batch = self._simulate_text_line(batch)'''

        '''process each image and apply agument if need'''
        res_imgs = [self.process_img(img) for img in batch.imgs]
        ''' calculate the maximum allowed text length based on the image width (shape[0] and [0] is the x in [x,y], divide by 4 assumming each image taking up 4 pixel horizontally)'''
        max_text_len = res_imgs[0].shape[0] // 4
        ''' trim the ground truth text to the max text length'''
        res_gt_texts = [self._truncate_label(gt_text, max_text_len) for gt_text in batch.gt_texts]
        return Batch(imgs=res_imgs, gt_texts=res_gt_texts, batch_size=batch.batch_size, img_paths=batch.img_paths)

