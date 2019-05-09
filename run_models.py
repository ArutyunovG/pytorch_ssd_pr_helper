from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from google.protobuf import text_format

import caffe
import tensorflow as tf

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

    def __eq__(self, other):
        return \
            self.label == other.label and \
            abs(self.confidence - other.confidence) < 0.01 and \
            all([abs(self.bbox[i] - other.bbox[i]) < 0.01 for i in range(4)])

    def __str__(self):
        return \
            str(self.label) + ' ' + str(self.confidence) + ' ' + \
            str(self.bbox[0]) + ' ' + str(self.bbox[1]) + ' ' + \
            str(self.bbox[2]) + ' ' + str(self.bbox[3])


def print_dets(dets, file_path):

    dets.sort(reverse = True)
    with open(file_path, 'w') as f:
        for i, det in enumerate(dets):
            if i > 0:
                f.write('\n')
            f.write(str(det))


def run_caffe_model(
        prototxt,
        caffemodel,
        mean
    ):

    img = cv2.imread('classic_300x300.bmp')

    inp_image = np.float32(img)

    inp_image[:, :, 0] -= mean[0]
    inp_image[:, :, 1] -= mean[1]
    inp_image[:, :, 2] -= mean[2]

    inp_image = np.transpose(inp_image, (2, 0, 1))
    inp_image.shape = (1,) +  inp_image.shape

    net = caffe.Net(
            prototxt,
            caffemodel,
            caffe.TEST
        )

    net.blobs['data'].data[...] = inp_image

    out = net.forward()
    detection_out = out['detection_out']

    assert len(detection_out.shape) == 4
    assert detection_out.shape[0] == 1
    assert detection_out.shape[1] == 1
    assert detection_out.shape[3] == 7

    dets = list()

    for i in range(detection_out.shape[2]):

        batch_idx = detection_out[0][0][i][0]

        assert batch_idx == 0

        label = int(detection_out[0][0][i][1] + 0.5)
        conf = detection_out[0][0][i][2]

        if conf < threshold:
            continue

        x1, y1, x2, y2 = detection_out[0][0][i][3:7]

        x1 *= inp_image.shape[3]
        y1 *= inp_image.shape[2]
        x2 *= inp_image.shape[3]
        y2 *= inp_image.shape[2]

        dets.append(Detection(
            confidence = conf,
            label = label,
            bbox = [x1, y1, x2, y2]
        ))

    print_dets(dets, 'caffe.txt')

def run_tf_model(
        pb_path
    ):
    
    img = cv2.imread('classic_600x600.bmp')

    inp_image = np.flip(img, axis = 2)
    inp_image.shape = (1,) +  inp_image.shape

    with tf.Session() as sess:

        graph_def = tf.GraphDef()
        with open(pb_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def)
        graph = tf.get_default_graph()

        inp_tensor = graph.get_tensor_by_name(
            'import/image_tensor:0')

        num_detections = graph.get_tensor_by_name(
            'import/num_detections:0')
        detection_classes = graph.get_tensor_by_name(
            'import/detection_classes:0')
        detection_boxes = graph.get_tensor_by_name(
            'import/detection_boxes:0')
        detection_scores = graph.get_tensor_by_name(
            'import/detection_scores:0')

        num_detections, \
        detection_classes, \
        detection_boxes, \
        detection_scores = \
            tuple(sess.run([num_detections, 
                  detection_classes, 
                  detection_boxes, 
                  detection_scores],
                  feed_dict = {inp_tensor: inp_image}))

        dets = list()
        for i in range(num_detections[0]):
            
            conf = detection_scores[0][i]

            if conf < threshold:
                continue

            label = int(detection_classes[0][i] + 0.5)
            y1, x1, y2, x2 = detection_boxes[0][i][:]

            x1 *= inp_image.shape[2]
            y1 *= inp_image.shape[1]
            x2 *= inp_image.shape[2]
            y2 *= inp_image.shape[1]

            dets.append(Detection(
                    confidence = conf,
                    label = label,
                    bbox = [x1, y1, x2, y2]
                ))

        print_dets(dets, 'tf.txt')

def run_caffe2_model(
              predict_path, 
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

    print('Running Caffe2 on SSD model')
    run_caffe2_model('ConvertedModels/Caffe2/SSD/ssd300x300_vgg.pbtxt',
              'ConvertedModels/Caffe2/SSD/ssd300x300_vgg.pb',
              'classic_300x300.bmp'
    )
    print('Done')

    print('Running Caffe2 on FRCNN model')
    run_caffe2_model('ConvertedModels/Caffe2/FRCNN/faster_rcnn_resnet50_coco_2018_01_28.pbtxt',
              'ConvertedModels/Caffe2/FRCNN/faster_rcnn_resnet50_coco_2018_01_28.pb',
              'classic_600x600.bmp'
    )
    print('Done')

    print('Running Caffe (reference framework) on SSD model')
    run_caffe_model(
        'ConvertedModels/Caffe/deploy.prototxt',
        'ConvertedModels/Caffe/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel',
        [104, 117, 123]
    )
    print('Done')
    

    print('Running TensorFlow (reference framework) on FRCNN model')
    run_tf_model(
        'ConvertedModels/TF/frozen_inference_graph.pb'
    )
    print('Done')
        
