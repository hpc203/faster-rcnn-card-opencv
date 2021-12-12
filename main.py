import argparse
import cv2

class faster_rcnn():
    def __init__(self, confThreshold=0.6):
        self.confThreshold = confThreshold
        self.net = cv2.dnn.readNet('faster_rcnn.pb', 'faster_rcnn.pbtxt')
    def detect(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame)
        self.net.setInput(blob)
        cvOut = self.net.forward()

        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            if score > self.confThreshold:
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
                cv2.putText(frame, 'card:'+str(round(score, 3)), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Use this script to run text detection deep learning networks using OpenCV.')
    parser.add_argument('--imgpath', type=str, default="test.jpg", help='Path to a image.')
    parser.add_argument('--confThreshold', type=float, default=0.6,
                        help="minimum probability to filter weak detections")
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = faster_rcnn(confThreshold=args.confThreshold)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()