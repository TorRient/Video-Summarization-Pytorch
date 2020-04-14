import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to video file")
args = parser.parse_args()

cap = cv2.VideoCapture(args.path)

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        count += 1
        length = 6 - len(str(count))
        name = 'test/frames/' + '0'*length + str(count) + '.jpg'
        cv2.imwrite(name, frame)

cap.release()