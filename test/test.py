
# plot photo with detected faces using opencv cascade classifier
import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle


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
        face = cv2.rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
        #print('{0} {1} {2} {3}'.format(x, y, width, height))
        print(box)
        img = cv2.GaussianBlur(img, (7,7), 0)
    cv2.imwrite(output_path, img)
    return True
    # show the image
    #imshow('face detection', pixels)
    # keep the window open until we press a key
    #waitKey(0)
    # close the window
    #destroyAllWindows()

if __name__ == '__main__':
    test('../res/evelyn-claire.webp')
    #pass
