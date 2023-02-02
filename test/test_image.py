# plot photo with detected faces using opencv cascade classifier
import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import sys

def test(input_path: str, output_path = '') -> bool:
    import os.path
    if not os.path.exists(input_path):
        print('{} not exists'.format(input_path))
        return False

    if not output_path:
        output_path = 'blurred.jpg'

    # load the photograph
    img = imread(input_path)
    # for more models: https://github.com/kipr/opencv/tree/master/data/haarcascades
    # load the pre-trained model
    classifier = CascadeClassifier('../res/models/haarcascades/haarcascade_frontalface_default.xml')
    # perform face detection
    bboxes = classifier.detectMultiScale(img)
    # print bounding box for each detected face
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        img = cv2.rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
        #print('{0} {1} {2} {3}'.format(x, y, width, height))
        print(box)
        # get face from image
        face = img[y:y2, x:x2]
        # perform Gaussian Blur
        blurred = cv2.GaussianBlur(face, (51,51), 0)
        # insert blurred face back into origianl image
        img[y:y2, x:x2] = blurred

        # show the image
        imshow('face', img)
        # keep the window open until we press a key
        waitKey(0)
        # close the window
        destroyAllWindows()

    cv2.imwrite(output_path, img)
    return True

def usage():
    print('''\
Usage:
\t* python3 test_image.py
\t* python3 test_image.py input_path
\t* python3 test_image.py input_path output_path
          ''')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        test('../res/evelyn-claire.webp')
    elif len(sys.argv) == 2:
        test(sys.argv[1])
    elif len(sys.argv) == 3:
        test(sys.argv[1], sys.argv[2])
    else:
        usage()
