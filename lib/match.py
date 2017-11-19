import cv2

def one_match(template, img_grey, return_dict,i):
    return_dict[i] = cv2.matchTemplate(img_grey, template, cv2.TM_SQDIFF_NORMED)