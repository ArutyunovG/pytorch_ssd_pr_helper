from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from google.protobuf import text_format

import cv2
import numpy as np

threshold = 0.5

class Detection(object):

    def __init__(self,
                 label,
                 confidence,
                 bbox):

        self.label = label
        self.confidence = confidence
        self.bbox = bbox

    def __lt__(self, other):

        return self.confidence < other.confidence

def print_dets(dets, file_path):

    dets.sort(reverse = True)
    with open(file_path, 'w') as f:
        for i, det in enumerate(dets):

            if i > 0:
                f.write('\n')

            f.write(str(det.label) + ' ' + str(det.confidence))
            
            assert len(det.bbox) == 4
            for coord in det.bbox:
                f.write(' ' + str(coord))


def run_caffe2_model(predict_path, 
              init_path,
              img_path):

    img = cv2.imread(img_path)

    inp_image = np.float32(img)
    inp_image = np.transpose(inp_image, (2, 0, 1))
    inp_image.shape = (1,) +  inp_image.shape


    predict_net = caffe2_pb2.NetDef()
    with open(predict_path, 'r') as f:
        text_format.Parse(f.read(), predict_net)

    init_net = caffe2_pb2.NetDef()
    with open(init_path, 'rb') as f:
        init_net.ParseFromString(f.read())

    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = workspace.GpuDeviceType
    device_option.device_id = 0
    init_net.device_option.CopyFrom(device_option)
    workspace.RunNetOnce(init_net)

    workspace.CreateNet(predict_net)

    workspace.FeedBlob('data', inp_image, device_option = core.DeviceOption(caffe2_pb2.CUDA, 0))

    im_info = np.ones(shape = (1, 3), dtype = np.float32)
    im_info[0, 0], im_info[0, 1] = inp_image.shape[2], inp_image.shape[3]
    workspace.FeedBlob('im_info', im_info, device_option = core.DeviceOption(caffe2_pb2.CPU, 0))

    workspace.RunNet(predict_net.name)

    score_nms = workspace.FetchBlob('score_nms')
    bbox_nms = workspace.FetchBlob('bbox_nms')
    class_nms = workspace.FetchBlob('class_nms')

    dets = list()

    for i in range(score_nms.shape[0]):

        conf = score_nms[i]

        if conf < threshold:
            continue

        x1, y1, x2, y2 = bbox_nms[i]

        dets.append(Detection(
                    confidence = conf,
                    label = int(class_nms[i] + 0.5),
                    bbox = [x1, y1, x2, y2]
                ))

    print_dets(dets, predict_net.name + '_caffe2.txt')


if __name__ == '__main__':

    run_caffe2_model('ConvertedModels/Caffe2/SSD/ssd300x300_vgg.pbtxt',
              'ConvertedModels/Caffe2/SSD/ssd300x300_vgg.pb',
              'classic_300x300.bmp'
    )

    run_caffe2_model('ConvertedModels/Caffe2/FRCNN/faster_rcnn_resnet50_coco_2018_01_28.pbtxt',
              'ConvertedModels/Caffe2/FRCNN/faster_rcnn_resnet50_coco_2018_01_28.pb',
              'classic_600x600.bmp'
    )
