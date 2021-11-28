import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
  config_file = 'Object Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # configuration file
  frozen_model = 'Object Detection/frozen_inference_graph.pb' # model
  model = cv2.dnn_DetectionModel(frozen_model, config_file) # loaded model in memory
  classLabels = [] # List of class Labels
  file_name = 'Object Detection/Labels.txt'
  with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

  print(classLabels)

  model.setInputSize(320,320) # input configuration file defined this as the input size
  model.setInputScale(1.0/127.5) # 255(all gray levels)/2
  model.setInputMean((127.5,127.5,127.5)) # mobilenet=>[-1, 1]
  model.setInputSwapRB(True) # input in RGB format to model
  # Read an image
  img = cv2.imread('Images/man_photo.jpg')
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # bgr format by default so convert to rgb format
  plt.show()
  # Giving input to the Mobilenet Model for detecting objects in the frame(or picture)
  ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.6) # returns those classes which have higher 
                    # x% confidence

  # Printing Result here

  # which all classes present in the frame ?
  for idx in ClassIndex:
    print(f'This class is present in frame:- {classLabels[idx-1]}')