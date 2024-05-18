import cv2
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from random import randrange

def resize(img, w, h):
    if img.shape[1] > w:
        img = cv2.resize(img, [w, img.shape[0]])

    if img.shape[0] > h:
        img = cv2.resize(img, [img.shape[1], h])
    return img

def harris_corner(img, dilated):
    # Convert cv2 Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the Grayscale image to float32
    gray = np.float32(gray)

    # Use cv2 harris corner function, img = input image, 2 = block size, 3 = ksize, 0.04 =
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    if dilated:
        # make the harris corner points larger to see them easier
        dst = cv2.dilate(dst, None)

    # if dst value is above 10 percent of the max dst value for all pixels then paint that pixel red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img

def sift_detector(img, flags, return_image):
    # Convert cv2 Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use cv2 to create a SIFT object
    sift = cv2.SIFT_create()
    # use sift object to detect key points in the grayscale image
    kp = sift.detect(gray, None)

    if flags: # use key points and the image to draw them onto the image with flags
        new_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else: # use key points and the image to draw them onto the image without flags
        new_img = cv2.drawKeypoints(img, kp, img)

    # use sift object to compute the descriptors
    kp, des = sift.compute(gray, kp)
    print(des[1])
    print(des[1].shape)

    # return img with keypoints drawn on, set of key points and set of descriptors
    if return_image:
        return new_img, kp, des
    else:
        return kp, des

def calculate_ssd(descriptor_X, descriptor_Y):
    descriptor_X = np.array(descriptor_X)
    descriptor_Y = np.array(descriptor_Y)

    squared_diff = (descriptor_X - descriptor_Y) ** 2

    sum_of_squared_diff = np.sum(squared_diff)

    return sum_of_squared_diff

def two_best_matcher(des_left, des_right):
    matches = []
    progress = 0
    one_percent = len(des_left)/100
    for i in range(len(des_left)): # for each query descriptor
        index_best = -1
        index_second_best = -1
        best_dist = float('inf')
        second_best_dist = float('inf')
        for j in range(len(des_right)): # check against all training descriptors
            # calculate ssd
            ssd = calculate_ssd(des_left[i], des_right[j])
            # if ssd is less than best dist
            if ssd < best_dist:
                # update second best dist and index
                second_best_dist = best_dist
                index_second_best = index_best

                # update best dist and index
                best_dist = ssd
                index_best = j
            # if ssd is greater than best but less than second best
            elif ssd > best_dist and ssd < second_best_dist:
                # update second best dist and index
                second_best_dist = ssd
                index_second_best = j
        progress = progress + 1
        if progress % one_percent == 0:
            print(f"{progress/one_percent} percent complete")
        # create best match object
        match_best = cv2.DMatch(int(i), int(index_best), float(best_dist))
        # create second best match object
        match_second_best = cv2.DMatch(int(i), int(index_second_best), float(second_best_dist))
        # add both objects to a tuple
        match_tuple = (match_best, match_second_best)
        # append tuple to matches
        matches.append(match_tuple)

    return matches

def draw_feature_matching(img_left, img_right):
    # Find keypoints and descriptors useing the sift_detector function
    img_left, kp_left, des_left = sift_detector(img_left, False, True)
    img_right, kp_right, des_right = sift_detector(img_right, False, True)

    # find matches using left as query descriptors and right as training descriptors
    matches = two_best_matcher(des_left, des_right)

    # Apply ratio test
    good = []
    Threshold = 0.7
    for m in matches:
        if m[0].distance / m[1].distance < Threshold:
            good.append([m[0]])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img3

def stitch(img, img_):

    kp1, des1 = sift_detector(img_, False, False)
    kp2, des2 = sift_detector(img, False, False)
    print(kp1.shape)
    print(des1.shape)
    print(kp2.shape)
    print(des2.shape)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = two_best_matcher(des2, des1)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    # cv.drawMatchesKnn expects list of lists as matches.
    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # print H
    else:
        raise AssertionError("Can’t find enough keypoints.")

    dst = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img
    return dst

def setup_frames(filename1, filename2):
    frame1 = cv2.imread(filename1)
    frame2 = cv2.imread(filename2)
    frame1 = resize(frame1, 700, 500)
    frame2 = resize(frame2, 700, 500)
    return frame1, frame2

def gui():
    sg.theme("LightGreen")

    # Define the window layout


    layout = [

        [   # Added text UI to show some title text
            sg.Text("COM31006: Salient feature, Matching and Image Stitch: Assignment", size=(60, 1), justification="center")
        ],

        [
            sg.Text("Image File 1"),
            sg.In(size=(40, 1), enable_events=True, key="-FILE-"),
            sg.FileBrowse(),
            sg.Text("Image File 2"),
            sg.In(size=(40, 1), enable_events=True, key="-FILE2-"),
            sg.FileBrowse(),
        ],

        [
            sg.Image(filename="", key="-IMAGE-"),
            sg.Image(filename="", key="-IMAGE2-")
        ],

        [sg.Radio("None", "Radio", True, size=(10, 1), key="-NORMAL-")],

        [
            sg.Radio("Harris Corner Detector", "Radio", size=(20, 1), key="-HARRIS CORNER-"),
            sg.Radio("Harris Corner Detector (dilated)", "Radio", size=(30, 1), key="-HARRIS CORNER DILATED-"),
            sg.Radio("SIFT feature point Detector", "Radio", size=(20, 1), key="-SIFT-"),
            sg.Radio("SIFT feature point Detector (Display flags)", "Radio", size=(40, 1), key="-FLAG-"),
        ],

        [sg.Radio("Brute Force Matcher", "Radio", size=(20, 1), key="-MATCHER-")],

        [sg.Radio("Brute Force stitch", "Radio", size=(20, 1), key="-STITCH-")],

        [sg.Button("Exit", size=(10, 1))],

    ]
    # Create the window and show it without the plot

    window = sg.Window("OpenCV Integration", layout, location=(800, 400))

    just_switched = False
    show_both = False
    filename1 = ""
    filename2 = ""
    prev = ""

    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-FILE-":
            filename1 = values ["-FILE-"]
            prev = ""
        elif event == "-FILE2-":
            filename2 = values["-FILE2-"]
            prev = ""

        if filename1 != "" and filename2 != "":

            if values["-NORMAL-"]:
                show_both = True

                if prev != "normal":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    just_switched = True
                    prev = "normal"

            elif values["-HARRIS CORNER-"]:
                show_both = True

                if prev != "harris corner":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    frame1 = harris_corner(frame1, False)
                    frame2 = harris_corner(frame2, False)
                    just_switched = True
                    prev = "harris corner"

            elif values["-HARRIS CORNER DILATED-"]:
                show_both = True

                if prev != "harris corner dilated":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    frame1 = harris_corner(frame1, True)
                    frame2 = harris_corner(frame2, True)
                    just_switched = True
                    prev = "harris corner dilated"

            elif values["-SIFT-"]:
                show_both = True

                if prev != "sift":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    frame1, kp, des = sift_detector(frame1, False, True)
                    frame2, kp, des = sift_detector(frame2, False, True)
                    just_switched = True
                    prev = "sift"

            elif values["-FLAG-"]:
                show_both = True

                if prev != "flag":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    frame1, kp, des = sift_detector(frame1, True, True)
                    frame2, kp, des = sift_detector(frame2, True, True)
                    just_switched = True
                    prev = "flag"

            elif values["-MATCHER-"]:
                show_both = False

                if prev != "matcher":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    frame1 = draw_feature_matching(frame1, frame2)
                    just_switched = True
                    prev = "matcher"

            elif values["-STITCH-"]:
                show_both = False

                if prev != "stitch":
                    frame1, frame2 = setup_frames(filename1, filename2)
                    frame1 = stitch(frame1, frame2)
                    just_switched = True
                    prev = "stitch"

        if just_switched and show_both:
            # if radio just switched and both frames should be shown,
            # resize frames and update them, make frame 2 visible
            frame1 = resize(frame1, 700, 500)
            frame2 = resize(frame2, 700, 500)
            imgbytes = cv2.imencode(".png", frame1)[1].tobytes()
            imgbytes2 = cv2.imencode(".png", frame2)[1].tobytes()

            window["-IMAGE-"].update(data=imgbytes)
            window["-IMAGE2-"].update(data=imgbytes2, visible=True)

            just_switched = False

        elif just_switched and not show_both:
            # if radio just switched and only frame 1 should be shown,
            # resize frame 1 and update it, make frame 2 not visible
            frame1 = resize(frame1, 1400, 500)
            imgbytes = cv2.imencode(".png", frame1)[1].tobytes()

            window["-IMAGE-"].update(data=imgbytes)
            window["-IMAGE2-"].update(visible=False)

            just_switched = False

    window.close()

if __name__ == '__main__':
    # harris_corner_detector("wall.png")
    # sift_detector("wall.png")
    gui()
    #left = cv2.imread("Images/left.png")
    #right = cv2.imread("Images/right.png")
    #brute_force_matching(left, right)