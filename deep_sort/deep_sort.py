import numpy as np
import torch
import sys

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

from scipy.stats import multivariate_normal
import numpy as np
import torch.nn as nn
import torchvision

sys.path.append('deep_sort/deep/reid')
#from torchreid.utils import FeatureExtractor

__all__ = ['DeepSort']


def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        #Outputs batch X 512 X 1 X 1 
        self.net = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(64,128,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            #nn.Dropout2d(p=0.4),            


            nn.Conv2d(128,256,kernel_size=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(256,256,kernel_size=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #nn.Dropout2d(p=0.4),    

            nn.Conv2d(256,512,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),    

            #1X1 filters to increase dimensions
            nn.Conv2d(512,1024,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),    

            )


    def forward_once(self, x):
        output = self.net(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        
        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2,input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1,output2,output3

        return output1, output2

class DeepSort(object):
    def __init__(self, wt_path, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):

        #self.extractor = FeatureExtractor(
        #    model_name=model_type,
        #    device=str(device)
        #)

        self.encoder = torch.load(wt_path)			
        self.encoder = self.encoder.cuda()
        self.encoder = self.encoder.eval()
        self.gaussian_mask = get_gaussian_mask().cuda()

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=False):
        self.height, self.width = ori_img.shape[:2]
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = np.array(bbox_tlwh)

        # generate detections
        processed_crops = self.pre_process(ori_img,detections).cuda()
        processed_crops = self.gaussian_mask * processed_crops
        
        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()
        
        if len(features.shape)==1:
            features = np.expand_dims(features,0)

        #features = self._get_features(bbox_xywh, ori_img)
        #bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = self.non_max_suppression(boxes, 0.85,scores) #get rid?
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            l = self.tracker.prev_track_clsses[track_id] 
            class_id = max(set(l), key = l.count)
            #class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def pre_process(self,frame,detections):	
        
        transforms = torchvision.transforms.Compose([ \
		torchvision.transforms.ToPILImage(),\
		torchvision.transforms.Resize((128,128)),\
		torchvision.transforms.ToTensor()])
        
        crops = []
        for d in detections:
            
            for i in range(len(d)):
                if d[i] <0:
                    d[i] = 0	
                    
            img_h,img_w,img_ch = frame.shape
            
            xmin,ymin,w,h = d
            
            if xmin > img_w:
                xmin = img_w
                
            if ymin > img_h:
                ymin = img_h
            
            xmax = xmin + w
            ymax = ymin + h
            
            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))
            
            try:
                crop = frame[ymin:ymax,xmin:xmax,:]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue
            
        crops = torch.stack(crops)
        
        return crops

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def non_max_suppression(self, boxes, max_bbox_overlap, scores=None):
        """Suppress overlapping detections.

        Original code from [1]_ has been adapted to include confidence score.

        .. [1] http://www.pyimagesearch.com/2015/02/16/
            faster-non-maximum-suppression-python/

        Examples
        --------

            >>> boxes = [d.roi for d in detections]
            >>> scores = [d.confidence for d in detections]
            >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
            >>> detections = [detections[i] for i in indices]

        Parameters
        ----------
        boxes : ndarray
            Array of ROIs (x, y, width, height).
        max_bbox_overlap : float
            ROIs that overlap more than this values are suppressed.
        scores : Optional[array_like]
            Detector confidence score.

        Returns
        -------
        List[int]
            Returns indices of detections that have survived non-maxima suppression.

        """
        if len(boxes) == 0:
            return []

        boxes = boxes.astype(np.float)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2] + boxes[:, 0]
        y2 = boxes[:, 3] + boxes[:, 1]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if scores is not None:
            idxs = np.argsort(scores)
        else:
            idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > max_bbox_overlap)[0])))

        return pick
