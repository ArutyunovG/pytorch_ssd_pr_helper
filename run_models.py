from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from google.protobuf import text_format

import cv2
import numpy as np

threshold = 0.5


def run_model(predict_path, 
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

    # clear file contents
    with open(predict_net.name + '_caffe2.txt', 'w') as f:
        pass

    line_idx = 0
    for i in range(score_nms.shape[0]):

        conf = score_nms[i]

        if conf < threshold:
            continue

        conf = int((conf * 100) + 0.5) / 100.0

        x1, y1, x2, y2 = bbox_nms[i]

        with open(predict_net.name + '_caffe2.txt', 'a') as f:

            if line_idx > 0:
                f.write('\n')
            line_idx += 1

            f.write(str(int(class_nms[i] + 0.5)) + ' ' + str(conf))
            f.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2))


if __name__ == '__main__':

    run_model('ConvertedModels/SSD/ssd300x300_vgg.pbtxt',
              'ConvertedModels/SSD/ssd300x300_vgg.pb',
              'classic_300x300.bmp'
    )

    run_model('ConvertedModels/FRCNN/faster_rcnn_resnet50_coco_2018_01_28.pbtxt',
              'ConvertedModels/FRCNN/faster_rcnn_resnet50_coco_2018_01_28.pb',
              'classic_600x600.bmp'
    )
