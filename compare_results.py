from run_models import Detection

def compare(file1, file2):
    
    lines1 = None
    lines2 = None

    with open(file1, 'r') as f:
        lines1 = f.readlines()

    with open(file2, 'r') as f:
        lines2 = f.readlines()  

    assert lines1 is not None 
    assert lines2 is not None 
    assert len(lines1) == len(lines2)

    for i in range(len(lines1)):

        label, confidence, x1, y1, x2, y2 = lines1[i].split(' ')
        det1 = Detection(
            label = int(float(label) + 0.5),
            confidence = float(confidence),
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        )

        label, confidence, x1, y1, x2, y2 = lines2[i].split(' ')
        det2 = Detection(
            label = int(float(label) + 0.5),
            confidence = float(confidence),
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        )

        try:
            assert det1 == det2
        except AssertionError:
            print('Comparison failed!\n')
            print('Detections were:\n')
            print(det1)
            print(det2)
            raise

if __name__ == '__main__':

    print('Comparing SSD results between Caffe2 and Caffe (reference framework)')
    compare('VGG_VOC0712_SSD_300x300_deploy_caffe2.txt',
            'caffe.txt')
    print('Comparison is successfull!')

    print('Comparing FRCNN results between Caffe2 and TensorFlow (reference framework)')
    compare('frozen_inference_graph_caffe2.txt',
            'tf.txt')
    print('Comparison is successfull!')      
