from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from typing import List, Optional, Union

warnings.filterwarnings("ignore", category=Warning)
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


class ErrorBase(Exception):
    '''
    Base class for Alg exceptions, bro.
    '''
    PREFIX = 'Error::'
    def __init__(self, message:str = ''):
        self.message = message
        super().__init__(self.message)

class InputError(ErrorBase):
    '''
    Errors for input incorrect.
    Including lenght, shape and dtype, etc. Sucker.
    '''
    def __init__(self, type:str, message:str = 'Input incorrect'):
        self.message = self.PREFIX + 'Type({}):'.format(type) + message
        super().__init__(self.message)

class ValueError(ErrorBase):
    '''
    Errors for output incorrect.
    Including out-of-range, not orthogonal, etc.
    '''
    def __init__(self, val:str, message:str = 'out of range.'):
        self.message = self.PREFIX + 'Val({}):'.format(val) + message
        super().__init__(self.message)


def argmax_2d(array:Optional[Union[list, np.array]], axis:int=0)->np.array:
    if not isinstance(array, (list, np.ndarray)):
        raise InputError('array', 'Type incorrect')
    if not isinstance(axis, int):
        raise InputError('axis', 'Type incorrect')
    if axis not in (0, 1):
        raise ValueError('axis', '{} invalid'.format(axis))

    if isinstance(array, list):
        array = np.array(array)

    if len(array.shape) != 2:
        raise ValueError('len(array.shape)', '{} are not 2'.format(len(array.shape)))

    max_array = np.amax(array, axis=axis)
    max_index = []
    length = len(array[0]) if axis == 0 else len(array)
    for i in range(length):
        value = array[:, i] if axis == 0 else array[i, :]
        candidates = np.argwhere(value == max_array[i])
        i = np.amax(candidates) # find the maximum of index
        max_index.append(i)

    max_index = np.array(max_index)
    return max_index

def check_thresh_value(thresh:float)->None:
    if not isinstance(thresh, float):
        raise InputError('thresh')
    if thresh < 0.0 or thresh > 1.0:
        raise ValueError('thresh {}'.format(thresh))

def check_confidence_thresh(thresh:Union[float, List[float]], num_classes:int)->List[float]:
    if isinstance(thresh, float):
        check_thresh_value(thresh)
        thresh = [thresh for _ in range(num_classes)]
    elif isinstance(thresh, list):
        if len(thresh) != num_classes:
            raise ValueError('confidence_thresh and num_classes', 'length are inconsistent')
        for t in thresh:
            check_thresh_value(t)
    else:
        raise InputError('thresh', 'Type invalid')

    return thresh

class TorchTensor:
    '''
    Utilities for statistics computed in tensor.
    Correct input type: Torch.tensor, Tensor
    '''
    @staticmethod
    def check_input_size_consistent(input:torch.Tensor, target:torch.Tensor)->None:
        if not isinstance(input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise InputError('input/target', 'Type error')
        if input.size() != target.size():
            raise InputError('Input size {} and target size {} inconsistent'.format(input.size(), target.size()))

    class Metric:
        class Point2D:

            @classmethod
            def euclidean_2d(cls, p1:TorchTensor.Metric.Point2D, p2:TorchTensor.Metric.Point2D)->torch.Tensor:
                return torch.sqrt(torch.square(p1.x-p2.x) + torch.square(p1.y-p2.y))

            def __init__(self, x:torch.IntTensor, y:torch.IntTensor):
                if not isinstance(x, (int, torch.Tensor)):
                    raise InputError('x', 'type incorrect')
                if not isinstance(y, (int, torch.Tensor)):
                    raise InputError('y', 'type incorrect')
                self.x = x.int() if isinstance(x, torch.Tensor) else torch.tensor(x)
                self.y = y.int() if isinstance(y, torch.Tensor) else torch.tensor(y)

        class Rect2D:
            @classmethod
            def closure(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->TorchTensor.Metric.Rect2D:
                xmin = torch.min(b1.xmin, b2.xmin)
                xmax = torch.max(b1.xmax, b2.xmax)
                ymin = torch.min(b1.ymin, b2.ymin)
                ymax = torch.max(b1.ymax, b2.ymax)

                closure_rect = TorchTensor.Metric.Rect2D(xmin, ymin, xmax, ymax)
                return closure_rect

            @classmethod
            def intersection(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->torch.Tensor:
                xmin = torch.max(b1.xmin, b2.xmin)
                xmax = torch.min(b1.xmax, b2.xmax)
                ymin = torch.max(b1.ymin, b2.ymin)
                ymax = torch.min(b1.ymax, b2.ymax)

                # if no intersection
                if xmin > xmax:
                    xmin = 0
                    xmax = 0
                if ymin > ymax:
                    ymin = 0
                    ymax = 0

                intersection = TorchTensor.Metric.Rect2D(xmin, ymin, xmax, ymax)
                return intersection.area()

            @classmethod
            def union(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->torch.Tensor:
                area1 = b1.area()
                area2 = b2.area()
                intersection = cls.intersection(b1, b2)
                union = area1 + area2 - intersection
                return union

            @classmethod
            def iou(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->torch.Tensor:
                iou = cls.intersection(b1, b2) / (cls.union(b1, b2) + 1e-12)
                return iou

            @classmethod
            def giou(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->torch.Tensor:
                closure_rect = cls.closure(b1, b2)
                area_closure = closure_rect.area()
                area_union = cls.union(b1, b2)
                giou = cls.iou(b1, b2) - (area_closure - area_union)/(area_closure + 1e-12)
                return giou

            @classmethod
            def diou(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->torch.Tensor:
                distance_center = TorchTensor.Metric.Point2D.euclidean_2d(b1.center(), b2.center())

                closure_rect = cls.closure(b1, b2)
                distance_diagonal = torch.sqrt(torch.square(closure_rect.width()) + torch.square(closure_rect.height()))

                penalty = torch.square(distance_center/(distance_diagonal + 1e-12))
                diou = cls.iou(b1, b2) - penalty
                return diou

            @classmethod
            def ciou(cls, b1:TorchTensor.Metric.Rect2D, b2:TorchTensor.Metric.Rect2D)->torch.Tensor:
                bias = 1e-12
                iou  = cls.iou(b1, b2)
                diou = cls.diou(b1, b2)
                ratio_b1 = b1.width() / (b1.height() + bias)
                ratio_b2 = b2.width() / (b2.height() + bias)

                # [Note] ratio of width and height
                # ratio: 0 - infinity
                # arctan(ratio): 0 - pi/2
                v = torch.square((2.0/torch.pi) * (torch.arctan(ratio_b1) - torch.arctan(ratio_b2)))
                alpha = v / (1 - iou + v + bias)
                ciou  = diou - alpha*v
                return ciou

            def __init__(self, xmin:torch.IntTensor, ymin:torch.IntTensor, xmax:torch.IntTensor, ymax:torch.IntTensor):
                if not isinstance(xmin, (int, torch.Tensor)):
                    raise InputError('xmin', 'type incorrect')
                if not isinstance(ymin, (int, torch.Tensor)):
                    raise InputError('ymin', 'type incorrect')
                if not isinstance(xmax, (int, torch.Tensor)):
                    raise InputError('xmax', 'type incorrect')
                if not isinstance(ymax, (int, torch.Tensor)):
                    raise InputError('ymax', 'type incorrect')
                if xmin > xmax:
                    raise ValueError('xmin {} > xmax {}'.format(xmin, xmax))
                if ymin > ymax:
                    raise ValueError('ymin {} > ymax {}'.format(ymin, ymax))

                self.xmin = xmin.int() if isinstance(xmin, torch.Tensor) else torch.tensor(xmin)
                self.ymin = ymin.int() if isinstance(ymin, torch.Tensor) else torch.tensor(ymin)
                self.xmax = xmax.int() if isinstance(xmax, torch.Tensor) else torch.tensor(xmax)
                self.ymax = ymax.int() if isinstance(ymax, torch.Tensor) else torch.tensor(ymax)

            def width(self)->torch.IntTensor:
                width = self.xmax - self.xmin
                if width < 0:
                    raise ValueError('width {} less than zero'.format(width))
                return width

            def height(self)->torch.IntTensor:
                height = self.ymax - self.ymin
                if height < 0:
                    raise ValueError('height {} less than zero'.format(height))
                return height

            def center(self)->TorchTensor.Metric.Point2D:
                x_center = int((self.xmin + self.xmax)/2)
                y_center = int((self.ymin + self.ymax)/2)
                return TorchTensor.Metric.Point2D(x_center, y_center)

            def top_left_corner(self)->TorchTensor.Metric.Point2D:
                return TorchTensor.Metric.Point2D(self.xmin, self.ymin)

            def bottom_right_corner(self)->TorchTensor.Metric.Point2D:
                return TorchTensor.Metric.Point2D(self.xmax, self.ymax)

            def area(self)->torch.Tensor:
                area = self.width()*self.height()
                if area < 0:
                    raise ValueError('area {} less than zero'.format(area))
                return area

            def shift(self, x_shift:torch.IntTensor=0, y_shift:torch.IntTensor=0):
                if isinstance(x_shift, (float, torch.FloatTensor)):
                    x_shift = torch.IntTensor(x_shift)
                if isinstance(y_shift, (float, torch.FloatTensor)):
                    y_shift = torch.IntTensor(y_shift)

                self.xmin += x_shift
                self.ymin += y_shift
                self.xmax += x_shift
                self.ymax += y_shift

            def scale(self, ratio:torch.FloatTensor):
                if ratio <= 0:
                    raise ValueError('ratio', 'less than 0')

                self.xmin = (self.xmin * ratio).to(torch.int)
                self.ymin = (self.ymin * ratio).to(torch.int)
                self.xmax = (self.xmax * ratio).to(torch.int)
                self.ymax = (self.ymax * ratio).to(torch.int)

            def limit_xmin_range(self, min:int, max:int):
                self.xmin = self._limit_range(self.xmin, min, max)

            def limit_xmax_range(self, min:int, max:int):
                self.xmax = self._limit_range(self.xmax, min, max)

            def limit_ymin_range(self, min:int, max:int):
                self.ymin = self._limit_range(self.ymin, min, max)

            def limit_ymax_range(self, min:int, max:int):
                self.ymax = self._limit_range(self.ymax, min, max)

            def _limit_range(self, value:int, min:int, max:int)->torch.IntTensor:
                if min >= max:
                    raise ValueError('Min value {} >= Max value {}'.format(min, max))
                ret = value
                if value < min:
                    ret = min
                elif value > max:
                    ret = max
                else:
                    ret = value

                if not isinstance(ret, torch.IntTensor):
                    ret = torch.tensor(ret).int()
                return ret

        class Bbox(Rect2D):

            def __init__(self, xmin:torch.IntTensor, ymin:torch.IntTensor, xmax:torch.IntTensor, ymax:torch.IntTensor,
                confidence:torch.FloatTensor=0.0, class_id:torch.IntTensor=-1):
                super().__init__(xmin, ymin, xmax, ymax)

                if not isinstance(confidence, (float, torch.Tensor)):
                    raise InputError('confidence', 'type incorrect')
                if not isinstance(class_id, (int, torch.Tensor)):
                    raise InputError('class_id', 'type incorrect')

                self.class_id = class_id.int() if isinstance(class_id, torch.Tensor) else torch.tensor(class_id)
                confidence = confidence if (confidence >= 0.0 and confidence <= 1.0) else 0.0
                self.confidence = confidence if isinstance(confidence, torch.Tensor) else torch.tensor(confidence)

        class Contour2D:

            @staticmethod
            def check_input_size_consistent(input:np.array, target:np.array)->None:
                if not isinstance(input, np.ndarray) or not isinstance(target, np.ndarray):
                    raise InputError('input/target', 'Type error')
                if input.shape != target.shape:
                    raise InputError('Input shape {} and target shape {} inconsistent'.format(input.shape, target.shape))

            @classmethod
            def intersection(cls, mask1:np.array, mask2:np.array)->float:
                cls.check_input_size_consistent(mask1, mask2)
                m1 = np.where(mask1 > 0, 1, mask1)
                m2 = np.where(mask2 > 0, 1, mask2)
                return np.sum(cv2.bitwise_and(m1, m2))

            @classmethod
            def union(cls, mask1:np.array, mask2:np.array)->float:
                cls.check_input_size_consistent(mask1, mask2)
                m1 = np.where(mask1 > 0, 1, mask1)
                m2 = np.where(mask2 > 0, 1, mask2)
                return np.sum(cv2.bitwise_or(m1, m2))

            @classmethod
            def iou(cls, c1:np.array, c2:np.array)->float:
                cls.check_input_size_consistent(c1, c2)
                iou = cls.intersection(c1, c2) / (cls.union(c1, c2) + 1e-12)
                return iou

            @classmethod
            def draw_contours(cls, image:np.array, contours:List[TorchTensor.Metric.Contour2D], color:tuple=(0, 255, 0), show_info:bool=False):
                img    = image.copy()
                canvas = image.copy()

                for contour in contours:
                    canvas[contour.mask()>0] = color
                    #cv2.drawContours(canvas, [contour.contour], -1, (0, 0, 0), thickness=1)

                for contour in contours:
                    if show_info:
                        x, y, w, h = contour.bbox()
                        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 0), 1)
                        cv2.putText(canvas, '{}x{}: {:.3f}'.format(w, h, contour.avg_conf), (x, y+h+15), \
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)

                alpha = 0.5
                result = cv2.addWeighted(img, alpha, canvas, 1-alpha, 0)
                return result

            def __init__(self, mask:np.array, contour:np.array, class_id:int, prob_map:np.array=None)->None:
                self.mask_ = mask # value: [0, 1] -> 1 for foreground, 0 for background
                self.contour = contour
                self.class_id = class_id
                self.avg_conf = self._calc_avg_conf(prob_map) if prob_map is not None else 0.0

            def mask(self, bg_color:int=0, fg_color:int=255)->np.array:
                foreground_mask = self.mask_ > 0
                background_mask = self.mask_ == 0
                ret = self.mask_.copy()
                ret[foreground_mask] = fg_color
                ret[background_mask] = bg_color
                return ret

            def area(self)->int:
                area = cv2.countNonZero(self.mask_)
                return area

            def bbox(self):
                x, y, w, h = cv2.boundingRect(self.contour)
                return x, y, w, h

            def _calc_avg_conf(self, prob_map:np.array)->float:
                mask = self.mask(bg_color=1, fg_color=0)
                masked_prob = np.ma.masked_array(prob_map, mask) # mask --> 1: hidden 0: show
                avg_conf = np.mean(masked_prob)
                return avg_conf

        @classmethod
        def euclidean(cls, x:torch.Tensor, y:torch.Tensor, w:torch.Tensor=None)->torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise InputError('x', 'Type incorrect')
            if not isinstance(y, torch.Tensor):
                raise InputError('y', 'Type incorrect')

            TorchTensor.check_input_size_consistent(x, y)
            if w is None:
                w = torch.ones(x.size())
            else:
                TorchTensor.check_input_size_consistent(x, w)

            d = torch.matmul(w, torch.square(x - y))
            return torch.sqrt(torch.sum(d))

        @classmethod
        def mahalanobis(cls, x:torch.Tensor, y:torch.Tensor, cov:torch.Tensor)->torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise InputError('x', 'Type incorrect')
            if not isinstance(y, torch.Tensor):
                raise InputError('y', 'Type incorrect')
            if not isinstance(cov, torch.Tensor):
                raise InputError('cov', 'Type incorrect')

            TorchTensor.check_input_size_consistent(x, y)
            if x.shape[0] != cov.shape[0]:
                raise InputError('Input size {} and target size {} inconsistent'.format(x.shape[0], cov.shape[0]))

            '''
            d(x, y) = sqrt( (x-y).T*cov^(-1)*(x-y) )
            '''
            delta = x - y
            m1 = torch.matmul(delta, torch.inverse(cov))
            m2 = torch.matmul(m1, delta)
            return torch.sqrt(m2)

        @classmethod
        def cosine_similarity(cls, x:torch.Tensor, y:torch.Tensor, dim:int=0)->torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise InputError('x', 'Type incorrect')
            if not isinstance(y, torch.Tensor):
                raise InputError('y', 'Type incorrect')

            TorchTensor.check_input_size_consistent(x, y)
            return torch.cosine_similarity(x, y, dim)

        @classmethod
        def accuracy(cls, input:torch.Tensor, target:torch.Tensor)->torch.Tensor:
            '''
            input.shape == 1 if input is the classs with maximum probability
            input.shape != 1 if input is the probability distribution
            '''
            if not isinstance(input, torch.Tensor):
                raise InputError('input', 'Type incorrect')
            if not isinstance(target, torch.Tensor):
                raise InputError('target', 'Type incorrect')

            preds = input if len(input.shape) == 1 else input.argmax(dim = 1)
            TorchTensor.check_input_size_consistent(preds, target)

            correct_num = preds.eq(target).sum().item()
            return float(correct_num) / float(len(target))

        class Validator:
            class Metric:
                def __init__(self) -> None:
                    self.tp = 0
                    self.fp = 0
                    self.fn = 0

            class MajorTicksLocator(matplotlib.ticker.LinearLocator):
                def tick_values(self, vmin:int, vmax:int):
                    if vmax < 100:
                        ticks = np.arange(vmin, vmax, 10).tolist()
                    elif vmax >= 100 and vmax < 200:
                        ticks = np.arange(vmin, vmax, 20).tolist()
                    elif vmax >= 200 and vmax < 400:
                        ticks = np.arange(vmin, vmax, 40).tolist()
                    elif vmax >= 400 and vmax < 800:
                        ticks = np.arange(vmin, vmax, 80).tolist()
                    elif vmax >= 800 and vmax < 1600:
                        ticks = np.arange(vmin, vmax, 160).tolist()
                    else:
                        ticks = np.arange(vmin, vmax, 500).tolist()
                    return ticks

            class MinorTicksLocator(matplotlib.ticker.LinearLocator):
                def tick_values(self, vmin:int, vmax:int):
                    if vmax < 100:
                        ticks = np.arange(vmin, vmax, 1).tolist()
                    elif vmax >= 100 and vmax < 200:
                        ticks = np.arange(vmin, vmax, 5).tolist()
                    elif vmax >= 200 and vmax < 400:
                        ticks = np.arange(vmin, vmax, 10).tolist()
                    elif vmax >= 400 and vmax < 800:
                        ticks = np.arange(vmin, vmax, 20).tolist()
                    else:
                        ticks = []
                    return ticks

            @abstractmethod
            def reset(self, *args, **kwargs):
                raise NotImplementedError('This function should be override')

            @abstractmethod
            def compute(self, *args, **kwargs):
                raise NotImplementedError('This function should be override')

            @abstractmethod
            def calc_metrics(self, *args, **kwargs):
                raise NotImplementedError('This function should be override')

        class ImageLevelValidator(Validator):
            def __init__(self, classes:int, confidence_thresh:Union[float, List[float]]) -> None:
                if not isinstance(classes, (int, torch.Tensor)):
                    raise InputError('classes', 'type incorrect')
                if classes <= 0:
                    raise ValueError('classes less than 0')

                self.num_classes = classes.int() if isinstance(classes, torch.torch.Tensor) else torch.tensor(classes)
                self.reset(confidence_thresh)

            def reset(self, confidence_thresh:Union[float, List[float]]):
                confidence_thresh = check_confidence_thresh(confidence_thresh, self.num_classes)
                self.confidence_thresh = confidence_thresh

                self.tp = 0
                self.total = 0
                self.accuracy = 0.0
                self.metrics_per_img = []

                self.tp_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.total_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.accuracy_per_class = np.zeros(self.num_classes)

            def compute(self, prob:torch.Tensor, target:torch.Tensor):
                # prob: [batch_size, num_classes]
                # target: [batch_size]
                if not isinstance(prob, torch.Tensor):
                    raise InputError('prob', 'type incorrect')
                if not isinstance(target, torch.Tensor):
                    raise InputError('target', 'type incorrect')
                if prob.shape[1] != self.num_classes:
                    raise InputError('prob.shape[1]', 'inconsistent with num_classes')
                if any(t >= self.num_classes for t in target):
                    raise ValueError('target', 'label >= num_classes')

                prob_max, class_id = prob.max(dim = 1) # [batch_size]
                TorchTensor.check_input_size_consistent(class_id, target)

                batch_size = len(target)
                for img_idx in range(batch_size):
                    if prob_max[img_idx] < self.confidence_thresh[class_id[img_idx]]:
                        class_id[img_idx] = -1

                    metric = self.Metric()
                    if class_id[img_idx] == target[img_idx]:
                        metric.tp += 1
                    else:
                        metric.fp += 1
                    self.metrics_per_img.append(metric)

                self.tp += torch.sum(class_id.eq(target))
                self.total += batch_size

                for idx in range(self.num_classes):
                    indexes = (target == idx)
                    total = indexes.sum().item()
                    tp = torch.sum(class_id[indexes].eq(target[indexes])).item()
                    self.tp_per_class[idx] += tp
                    self.total_per_class[idx] += total

            def calc_metrics(self):
                bias = 1e-12
                self.accuracy  = self.tp / float(self.total + bias)

                for class_id in range(self.num_classes):
                    self.accuracy_per_class[class_id] += (float(self.tp_per_class[class_id]) / float(self.total_per_class[class_id] + bias))

        class BboxLevelValidator(Validator):

            class PR:
                def __init__(self, class_id:torch.IntTensor, prob:torch.FloatTensor, found:torch.bool) -> None:
                    if not isinstance(class_id, (int, torch.Tensor)):
                        raise InputError('class_id', 'type incorrect')
                    if not isinstance(prob, (float, torch.Tensor)):
                        raise InputError('prob', 'type incorrect')
                    if class_id < 0:
                        raise ValueError('class_id', 'less than 0')
                    if prob < 0.0 or prob > 1.0:
                        raise ValueError('prob', 'out-of-range [0.0. 1.0]')

                    self.recall = 0.0
                    self.precision = 0.0
                    self.tp = 0
                    self.fp = 0
                    self.prob = prob if isinstance(prob, torch.Tensor) else torch.tensor(prob)
                    self.found = found
                    self.class_id = class_id.int() if isinstance(class_id, torch.Tensor) else torch.tensor(class_id)

            def __init__(self, classes:torch.IntTensor, confidence_thresh:Union[float, List[float]], iou_criterion, iou_thresh:float=0.25) -> None:
                if not isinstance(classes, (int, torch.Tensor)):
                    raise InputError('classes', 'type incorrect')
                if classes <= 0:
                    raise ValueError('classes less than 0')

                self.num_classes = classes.int() if isinstance(classes, torch.Tensor) else torch.tensor(classes)
                self.iou_criterion = iou_criterion
                self.reset(confidence_thresh, iou_thresh)

            def reset(self, confidence_thresh:Union[float, List[float]], iou_thresh:float=0.25) -> None:
                confidence_thresh = check_confidence_thresh(confidence_thresh, self.num_classes)
                check_thresh_value(iou_thresh)

                self.confidence_thresh = confidence_thresh
                self.iou_thresh = iou_thresh

                self.bboxs = []
                self.metrics_per_img = []

                self.tp = 0
                self.fp = 0
                self.fn = 0
                self.gt = 0
                self.avg_iou   = 0.0
                self.recall    = 0.0
                self.precision = 0.0
                self.f1_score  = 0.0
                self.map = 0.0

                self.tp_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.fp_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.fn_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.gt_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.recall_per_class = np.zeros(self.num_classes)
                self.precision_per_class = np.zeros(self.num_classes)
                self.f1_score_per_class  = np.zeros(self.num_classes)
                self.ap_per_class = np.zeros(self.num_classes)

            def compute(self, input:List[TorchTensor.Metric.Bbox], target:List[TorchTensor.Metric.Bbox], sorted:bool=False):
                '''
                1. input is the list of prediction bbox after nms preprocess (already sorted by objectness from high to low)
                2. check prediction bbox and ground truth bbox is matched (thresh, iou_thresh, class_id)
                '''
                if sorted:
                    pred = self._sort_by_confidence(input)
                else:
                    pred = input

                gt = len(target)
                self.gt += gt
                gt_per_class = [0 for _ in range(self.num_classes)]
                for t in target:
                    class_id = t.class_id
                    gt_per_class[class_id] += 1
                    self.gt_per_class[class_id] += 1

                truth_index_list = []
                metric = [self.Metric() for _ in range(self.num_classes)]
                for pred_bbox in pred:
                    prob = pred_bbox.confidence
                    class_id = pred_bbox.class_id

                    max_iou = 0.0
                    truth_index = -1
                    truth_class_id = -1

                    #print('===============================================================')
                    for gt_index, gt_bbox in enumerate(target):
                        current_iou = self.iou_criterion(pred_bbox, gt_bbox)
                        if current_iou > max_iou:
                            max_iou = current_iou
                            truth_index = gt_index
                            truth_class_id = gt_bbox.class_id

                    #print(prob, max_iou, truth_index, truth_index_list)
                    #print('===============================================================')
                    if (max_iou >= self.iou_thresh) and (truth_index > -1) and truth_index not in truth_index_list:
                        truth_index_list.append(truth_index)

                        if class_id == truth_class_id:
                            if prob >= self.confidence_thresh[class_id]:
                                self.avg_iou += max_iou
                                self.tp += 1
                                self.tp_per_class[class_id] += 1
                                metric[class_id].tp +=1
                            self.bboxs.append(self.PR(class_id, prob, found=True))
                        else:
                            if prob >= self.confidence_thresh[class_id]:
                                self.fp += 1
                                self.fp_per_class[class_id] += 1
                                metric[class_id].fp +=1
                            self.bboxs.append(self.PR(class_id, prob, found=False))
                    else:
                        if prob >= self.confidence_thresh[class_id]:
                            self.fp += 1
                            self.fp_per_class[class_id] += 1
                            metric[class_id].fp +=1
                        self.bboxs.append(self.PR(class_id, prob, found=False))

                for i in range(self.num_classes):
                    metric[i].fn = gt_per_class[i] - metric[i].tp

                self.metrics_per_img.append(metric)

            def calc_metrics(self, save_dir:str=None, epoch:int=None)->None:
                bias = 1e-12
                # overall
                self.fn        = self.gt - self.tp
                self.recall    = float(self.tp) / float(self.tp + self.fn + bias)
                self.precision = float(self.tp) / float(self.tp + self.fp + bias)
                self.f1_score  = (2 * self.recall * self.precision) / (self.recall + self.precision + bias)
                if self.tp > 0:
                    self.avg_iou /= self.tp

                # classes
                for class_id in range(self.num_classes):
                    self.fn_per_class[class_id]        = self.gt_per_class[class_id] - self.tp_per_class[class_id]
                    self.recall_per_class[class_id]    = float(self.tp_per_class[class_id]) / float(self.gt_per_class[class_id] + bias)
                    self.precision_per_class[class_id] = float(self.tp_per_class[class_id]) / float(self.tp_per_class[class_id] + self.fp_per_class[class_id] + bias)
                    self.f1_score_per_class[class_id]  = (2 * self.recall_per_class[class_id] * self.precision_per_class[class_id]) / float(self.recall_per_class[class_id] + self.precision_per_class[class_id] + bias)
                    self.ap_per_class[class_id]        = self._calc_average_precision(self.bboxs, class_id, save_dir, epoch)
                    if save_dir:
                        bbox_histogram = self._generate_bbox_histogram(self.bboxs, class_id)
                        self._draw_bbox_histogram(bbox_histogram, class_id, save_dir, epoch)

                self.map = float(np.mean(self.ap_per_class))

            def _calc_average_precision(self, bboxs:List[PR], class_id:int, save_dir:str=None, epoch:int=None)->torch.FloatTensor:
                # Search bboxs by class_id
                bboxs_sorted = [bbox for bbox in bboxs if bbox.class_id == class_id]
                bboxs_sorted.sort(key = lambda b: b.prob, reverse=True)

                pr_curve = [[0.0, 1.0]]
                pr_estimation_curve = []

                bias = 1e-12
                # Calculate cumulative tp, fp, recall and precision
                for i in range(len(bboxs_sorted)):
                    if i == 0:
                        if bboxs_sorted[i].found:
                            bboxs_sorted[i].tp += 1
                        else:
                            bboxs_sorted[i].fp += 1
                    else:
                        if bboxs_sorted[i].found:
                            bboxs_sorted[i].tp = bboxs_sorted[i-1].tp + 1
                            bboxs_sorted[i].fp = bboxs_sorted[i-1].fp
                        else:
                            bboxs_sorted[i].fp = bboxs_sorted[i-1].fp + 1
                            bboxs_sorted[i].tp = bboxs_sorted[i-1].tp

                    bboxs_sorted[i].recall    = float(bboxs_sorted[i].tp) / float(self.gt_per_class[class_id] + bias)
                    bboxs_sorted[i].precision = float(bboxs_sorted[i].tp) / float(bboxs_sorted[i].tp + bboxs_sorted[i].fp + bias)
                    #print('bbox {}: {:.3f} {} {:.3f} {:.3f}'.format(i, bboxs_sorted[i].prob.item(), bboxs_sorted[i].found, bboxs_sorted[i].recall, bboxs_sorted[i].precision))
                    pr_curve.append([bboxs_sorted[i].recall, bboxs_sorted[i].precision])

                # Calculate area of pr-cruve
                ap = 0.0
                max_precision = 0.0
                bboxs_sorted.insert(0, self.PR(class_id, 1.0, True)) # start point
                for i in reversed(range(1, len(bboxs_sorted))): # from end to begin
                    delta_recall = float(bboxs_sorted[i].recall - bboxs_sorted[i-1].recall)
                    if bboxs_sorted[i].precision > max_precision:
                        max_precision = bboxs_sorted[i].precision

                    ap += (max_precision*delta_recall)
                    pr_estimation_curve.append([bboxs_sorted[i].recall, max_precision])
                    #print(i, ap, delta_recall, max_precision)

                # Draw PR cruve
                if save_dir:
                    self._draw_pr_curve(class_id, ap, pr_curve, pr_estimation_curve, save_dir, epoch)

                return ap

            def _draw_pr_curve(self, class_id:int, ap:float, pr_curve:list, pr_estimation_curve:list, save_dir:str, epoch:int)->None:
                if save_dir is None:
                    return

                os.makedirs(save_dir, exist_ok=True)

                if epoch:
                    filename = 'pr-curve-class{}-ep{}.png'.format(class_id, epoch)
                    title = 'PR Curve\nClass{} EP{}'.format(class_id, epoch)
                else:
                    filename = 'pr-curve-class{}.png'.format(class_id)
                    title = 'PR Curve\nClass{}'.format(class_id)
                filename = os.path.join(save_dir, filename)

                fig, ax = plt.subplots(1, 1)
                ax.grid()
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.plot([pr[0] for pr in pr_curve]           , [pr[1] for pr in pr_curve], '--b')
                ax.plot([pr[0] for pr in pr_estimation_curve], [pr[1] for pr in pr_estimation_curve], 'r')
                ax.text(0.05, 0.05, 'AP:{:.3f}'.format(ap), bbox={'facecolor': 'red', 'alpha': 1.0, 'pad': 5}, color='black', fontsize=18)
                plt.title(title)
                plt.xlim(0.0, 1.05)
                plt.ylim(0.0, 1.05)
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
                plt.savefig(filename)
                plt.close(fig)

            def _generate_bbox_histogram(self, bboxs:List[PR], class_id:int)->None:
                # Search bboxs by class_id
                bboxs_sorted = [bbox for bbox in bboxs if bbox.class_id == class_id]
                bboxs_sorted.sort(key = lambda b: b.prob, reverse=True)

                histogram = {
                    'postive_samples' : [], # tp + fn
                    'negative_samples': []  # fp
                }

                for bbox in bboxs_sorted:
                    prob = bbox.prob.item()
                    if bbox.found:
                        histogram['postive_samples'].append(prob) # tp
                    else:
                        histogram['negative_samples'].append(prob) # fp

                fn = self.gt_per_class[class_id] - len(histogram['postive_samples'])
                if fn > 0:
                    histogram['postive_samples'].extend([0.0 for _ in range(fn)]) # fn

                return histogram

            def _draw_bbox_histogram(self, histogram:dict, class_id:int, save_dir:str, epoch:int)->None:
                if save_dir is None:
                    return

                os.makedirs(save_dir, exist_ok=True)

                if epoch: # train
                    filename = 'bbox-histogram-class{}-ep{}.png'.format(class_id, epoch)
                    title = 'Bbox Histogram\nClass{} EP{}'.format(class_id, epoch)
                else: # test
                    filename = 'bbox-histogram-class{}.png'.format(class_id)
                    title = 'Bbox Histogram\nClass{}'.format(class_id)
                filename = os.path.join(save_dir, filename)

                fig, ax = plt.subplots(1, 1)
                ax.grid()
                plt.hist(
                        x=[histogram['postive_samples'], histogram['negative_samples']],
                        label=['postive', 'negative'],
                        bins=10,
                        range=(0.0, 1.0)
                    )
                plt.title(title)
                plt.xlim(xmin=0.0, xmax=1.0)
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Amount")
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
                ax.yaxis.set_major_locator(self.MajorTicksLocator())
                ax.yaxis.set_minor_locator(self.MinorTicksLocator())
                plt.legend(loc='upper left')
                plt.savefig(filename)
                plt.close(fig)

            def _sort_by_confidence(self, input:List[TorchTensor.Metric.Bbox])->list:
                return sorted(input, key = lambda bbox : bbox.confidence, reverse=True)

        class ContourLevelValidator(Validator):

            class PR:
                def __init__(self, class_id:torch.IntTensor, prob:torch.FloatTensor, found:torch.bool) -> None:
                    if not isinstance(class_id, (int, torch.Tensor)):
                        raise InputError('class_id', 'type incorrect')
                    if class_id < 0:
                        raise ValueError('class_id', 'less than 0')
                    if prob < 0.0 or prob > 1.0:
                        raise ValueError('prob', 'out-of-range [0.0. 1.0]')
                    self.prob = prob
                    self.found = found
                    self.class_id = class_id.int() if isinstance(class_id, torch.Tensor) else torch.tensor(class_id)

            def __init__(self, classes:int, confidence_thresh:float, iou_thresh:float) -> None:
                if not isinstance(classes, (int, torch.Tensor)):
                    raise InputError('classes', 'type incorrect')
                if classes <= 0:
                    raise ValueError('classes', 'less than 0')

                self.num_classes = classes.int() if isinstance(classes, torch.Tensor) else torch.tensor(classes)
                self.reset(confidence_thresh, iou_thresh)

            def reset(self, confidence_thresh:Union[List[float], float], iou_thresh:float=0.45):
                confidence_thresh = check_confidence_thresh(confidence_thresh, self.num_classes)
                check_thresh_value(iou_thresh)

                self.confidence_thresh = confidence_thresh
                self.iou_thresh = iou_thresh

                self.contours = []
                self.metrics_per_img = []

                self.tp = 0
                self.fp = 0
                self.fn = 0
                self.gt = 0
                self.avg_iou   = 0.0
                self.recall    = 0.0
                self.precision = 0.0
                self.f1_score  = 0.0

                self.tp_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.fp_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.fn_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.gt_per_class = np.zeros(self.num_classes, dtype=np.int32)
                self.recall_per_class = np.zeros(self.num_classes)
                self.precision_per_class = np.zeros(self.num_classes)
                self.f1_score_per_class  = np.zeros(self.num_classes)

            def compute(self, input:List[TorchTensor.Metric.Contour2D], target:List[TorchTensor.Metric.Contour2D], sorted:bool=False):
                if sorted:
                    pred = self._sort_by_confidence(input)
                else:
                    pred = input

                gt = len(target)
                self.gt += gt
                gt_per_class = [0 for _ in range(self.num_classes)]
                for t in target:
                    class_id = t.class_id
                    gt_per_class[class_id] += 1
                    self.gt_per_class[class_id] += 1

                truth_index_list = []
                metric = [self.Metric() for _ in range(self.num_classes)]
                for pred_contour in pred:
                    prob = pred_contour.avg_conf
                    class_id = pred_contour.class_id

                    max_iou = 0.0
                    truth_index = -1
                    truth_class_id = -1
                    pred_mask = pred_contour.mask()
                    #print('===============================================================')
                    for gt_index, gt_contour in enumerate(target):
                        gt_mask   = gt_contour.mask()
                        curr_iou = TorchTensor.Metric.Contour2D.iou(pred_mask, gt_mask)
                        if curr_iou > max_iou:
                            max_iou = curr_iou
                            truth_index = gt_index
                            truth_class_id = gt_contour.class_id

                    #print('===============================================================')
                    if max_iou >= self.iou_thresh and truth_index > -1:

                        if truth_index in truth_index_list and class_id == truth_class_id:
                            continue

                        truth_index_list.append(truth_index)

                        if class_id == truth_class_id:
                            if prob >= self.confidence_thresh[class_id]:
                                self.avg_iou += max_iou
                                self.tp += 1
                                self.tp_per_class[class_id] += 1
                                metric[class_id].tp += 1
                            self.contours.append(self.PR(class_id, prob, found=True))
                        else:
                            if prob >= self.confidence_thresh[class_id]:
                                self.fp += 1
                                self.fp_per_class[class_id] += 1
                                metric[class_id].fp += 1
                            self.contours.append(self.PR(class_id, prob, found=False))
                    else:
                        if prob >= self.confidence_thresh[class_id]:
                            self.fp += 1
                            self.fp_per_class[class_id] += 1
                            metric[class_id].fp += 1
                        self.contours.append(self.PR(class_id, prob, found=False))

                for i in range(self.num_classes):
                    metric[i].fn = gt_per_class[i] - metric[i].tp

                self.metrics_per_img.append(metric)

            def calc_metrics(self, save_dir:str=None, epoch:int=None)->None:
                bias = 1e-12
                # overall
                self.fn        = self.gt - self.tp
                self.recall    = float(self.tp) / float(self.tp + self.fn + bias)
                self.precision = float(self.tp) / float(self.tp + self.fp + bias)
                self.f1_score  = (2 * self.recall * self.precision) / (self.recall + self.precision + bias)
                if self.tp > 0:
                    self.avg_iou /= self.tp

                # classes
                for class_id in range(1, self.num_classes): # exclude background (class 0)
                    self.fn_per_class[class_id]        = self.gt_per_class[class_id] - self.tp_per_class[class_id]
                    self.recall_per_class[class_id]    = float(self.tp_per_class[class_id]) / float(self.gt_per_class[class_id] + bias)
                    self.precision_per_class[class_id] = float(self.tp_per_class[class_id]) / float(self.tp_per_class[class_id] + self.fp_per_class[class_id] + bias)
                    self.f1_score_per_class[class_id]  = (2 * self.recall_per_class[class_id] * self.precision_per_class[class_id]) / float(self.recall_per_class[class_id] + self.precision_per_class[class_id] + bias)
                    if save_dir:
                        histogram = self._generate_histogram(self.contours, class_id)
                        self._draw_histogram(histogram, class_id, save_dir, epoch)

            def _sort_by_confidence(self, input:List[TorchTensor.Metric.Contour2D])->list:
                return sorted(input, key = lambda contour : contour.avg_conf, reverse=True)

            def _generate_histogram(self, contours:list, class_id:int)->None:
                # Search bboxs by class_id
                contours_sorted = [contour for contour in contours if contour.class_id == class_id]
                contours_sorted.sort(key = lambda b: b.prob, reverse=True)

                histogram = {
                    'postive_samples' : [], # tp + fn
                    'negative_samples': []  # fp
                }

                for contour in contours_sorted:
                    prob = contour.prob.item()
                    if contour.found:
                        histogram['postive_samples'].append(prob) # tp
                    else:
                        histogram['negative_samples'].append(prob) # fp

                fn = self.gt_per_class[class_id] - len(histogram['postive_samples'])
                if fn > 0:
                    histogram['postive_samples'].extend([0.0 for _ in range(fn)]) # fn

                return histogram

            def _draw_histogram(self, histogram:dict, class_id:int, save_dir:str, epoch:int)->None:
                if save_dir is None:
                    return

                os.makedirs(save_dir, exist_ok=True)

                if epoch: # train
                    filename = 'contour-histogram-class{}-ep{}.png'.format(class_id, epoch)
                    title = 'Contour Histogram\nClass{} EP{}'.format(class_id, epoch)
                else: # test
                    filename = 'contour-histogram-class{}.png'.format(class_id)
                    title = 'Contour Histogram\nClass{}'.format(class_id)
                filename = os.path.join(save_dir, filename)

                fig, ax = plt.subplots(1, 1)
                ax.grid()
                plt.hist(
                        x=[histogram['postive_samples'], histogram['negative_samples']],
                        label=['postive', 'negative'],
                        bins=10,
                        range=(0.0, 1.0)
                    )
                plt.title(title)
                plt.xlim(xmin=0.0, xmax=1.0)
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Amount")
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
                ax.yaxis.set_major_locator(self.MajorTicksLocator())
                ax.yaxis.set_minor_locator(self.MinorTicksLocator())
                plt.legend(loc='upper left')
                plt.savefig(filename)
                plt.close(fig)

        @classmethod
        def confusion_matrix(cls, input:torch.Tensor, target:torch.Tensor, cls_num:int = None)->torch.Tensor:
            '''
            both input and target are tensor with dim = (Num, )
            '''
            # Check input and target size
            TorchTensor.check_input_size_consistent(input, target)

            # Compute pred-input stack and instantiate matrix
            with torch.no_grad():
                stacked = torch.stack((target, input), dim = 1)
                # Check class numbers
                max_col = int(stacked.max().item()) + 1
                if not cls_num:
                    cls_num = max_col
                else:
                    if cls_num < max_col:
                        cls_num = max_col

                cmt = torch.zeros(cls_num, cls_num, dtype = torch.int16, requires_grad = False)

                for pair in stacked:
                    tl, pl = int(pair[0].item()), int(pair[1].item())
                    cmt[tl, pl] = cmt[tl, pl] + 1

                return cmt


if __name__ == '__main__':
    num_classes = 1
    validator = TorchTensor.Metric.BboxLevelValidator(classes=num_classes, confidence_thresh=0.1, iou_thresh=0.5, iou_criterion=TorchTensor.Metric.Bbox.iou)
    #=======================================================
    pred = []
    torch.manual_seed(5487)
    for _ in range(200):
        x = torch.randint(low=0, high=10, size=[1])
        y = torch.randint(low=0, high=10, size=[1])
        w = torch.randint(low=2, high=5, size=[1])
        h = torch.randint(low=2, high=5, size=[1])
        confidence = torch.rand(1)
        class_id = torch.randint(low=0, high=num_classes, size=[1])
        bbox = TorchTensor.Metric.Bbox(x, y, x + w, y + h, confidence, class_id)
        pred.append(bbox)

    pred = sorted(pred, key = lambda bbox : bbox.confidence, reverse=True)

    gt = []
    for _ in range(100):
        x = torch.randint(low=0, high=10, size=[1])
        y = torch.randint(low=0, high=10, size=[1])
        w = torch.randint(low=2, high=5, size=[1])
        h = torch.randint(low=2, high=5, size=[1])
        class_id = torch.randint(low=0, high=num_classes, size=[1])
        bbox = TorchTensor.Metric.Bbox(x, y, x + w, y + h, class_id=class_id)
        gt.append(bbox)
    #=======================================================
    validator.compute(pred, gt, sorted=False)
    validator.calc_metrics(save_dir=os.path.curdir)

    print(f'map:{validator.map}, \
        recall:{validator.recall}, \
        precision:{validator.precision}, \
        tp:{validator.tp}, \
        fp:{validator.fp}, \
        fn:{validator.fn}, \
        ap:{validator.ap_per_class[0]}')
