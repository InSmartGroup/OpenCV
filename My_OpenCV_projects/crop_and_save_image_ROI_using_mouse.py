import cv2

image = cv2.imread('sample.jpg', 1)
image_copy = image.copy()

press_down = ()
press_up = ()
roi = []


def draw_rectangle(event, x, y, flags, parameters):
    global press_down, press_up, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        press_down = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        press_up = (x, y)
        cv2.rectangle(image, press_down, press_up, (255, 255, 0), 2, lineType=cv2.LINE_8)
        roi = image_copy[press_down[1]:press_up[1], press_down[0]:press_up[0]]
        cv2.imwrite('Face_ROI.jpg', roi)
        press_down = ()
        press_up = ()
        roi = []


cv2.namedWindow("Image")
cv2.setMouseCallback('Image', draw_rectangle)

while True:

    cv2.putText(image, "Click & drag to draw, press 'C' to clear or ESC to quit", (10, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)

    if key == ord('C') or key == ord('c'):
        image = image_copy.copy()

    if key == 27:
        break

cv2.destroyAllWindows()
