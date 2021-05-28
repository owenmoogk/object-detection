import cv2

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

# getting all the objects into a list
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
	classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

detectionAnimals = ['bird', 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']

while True:
	success, img = capture.read()

	classIds, confs, bbox = net.detect(img, confThreshold=0.55)

	if len(classIds) != 0:
		for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
			if classNames[classId-1] in detectionAnimals:
				cv2.rectangle(img, box, color=(255,0,0), thickness=3)
				cv2.putText(img, classNames[classId-1] + str(confidence)[0:5], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)

	cv2.imshow('Output', img)

	cv2.waitKey(1)