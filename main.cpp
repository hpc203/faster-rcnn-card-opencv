#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class faster_rcnn
{
public:
	faster_rcnn(float confThreshold)
	{
		this->confThreshold = confThreshold;
		this->net = readNet("faster_rcnn.pb", "faster_rcnn.pbtxt");
	}
	void detect(Mat& frame);
private:
	float confThreshold;
	Net net;
};

void faster_rcnn::detect(Mat& frame)
{
	Mat blob = blobFromImage(frame);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	vector<float> confidences;
	vector<Rect> boxes;
	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* pdata = (float*)outs[i].data;
		int num_proposal = outs[i].size[2];
		int len = outs[i].size[3];
		for (int n = 0; n < num_proposal; n++)
		{
			const float score = pdata[2];
			if (score > this->confThreshold)
			{
				const int left = int(pdata[3] * frame.cols);
				const int top = int(pdata[4] * frame.rows);
				const int right = int(pdata[5] * frame.cols);
				const int bottom = int(pdata[6] * frame.rows);
				confidences.push_back(score);
				boxes.push_back(Rect(left, top, right - left, bottom - top));
			}
			pdata += len;
		}
	}

	for (int i = 0; i < boxes.size(); i++)
	{
		//Draw a rectangle displaying the bounding box
		rectangle(frame, Point(boxes[i].x, boxes[i].y), Point(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height), Scalar(0, 0, 255), 2);

		//Get the label for the class name and its confidence
		string label = format("%.2f", confidences[i]);
		label = "card:" + label;

		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int top = max(boxes[i].y, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(boxes[i].x, top - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
	}
	//imwrite("out.jpg", frame);
}

int main()
{
	faster_rcnn net(0.6);
	string imgpath = "test.jpg";
	Mat srcimg = imread(imgpath);
	net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}