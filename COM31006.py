import cv2
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from random import randrange

def resize(img, x, y):
    if img.shape[0] > x:
        img = cv2.resize(img, [x, img.shape[1]])

    if img.shape[1] > y:
        img = cv2.resize(img, [img.shape[0], y])
    return img

def harris_corner(img):
    # Convert cv2 Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the Grayscale image to float32
    gray = np.float32(gray)

    # Use cv2 harris corner function, img = input image, 2 = block size, 3 = ksize, 0.04 =
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # if dst value is above 10 percent of the max dst value for all pixels then paint that pixel red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    img = resize(img, 700, 500)
    return img

def harris_corner_dilated(img):
    # Convert cv2 Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the Grayscale image to float32
    gray = np.float32(gray)

    # Use cv2 harris corner function, img = input image, 2 = block size, 3 = ksize, 0.04 =
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # if dst value is above 10 percent of the max dst value for all pixels then paint that pixel red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    img = resize(img, 700, 500)
    return img

def sift_detector(img, flags):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    if flags:
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        img = cv2.drawKeypoints(gray, kp, img)

    kp, des = sift.compute(gray, kp)

    img = resize(img, 700, 500)
    return img, kp, des

def brute_force_matching(img1, img2):
    #img1 = cv2.imread('Images/left.png', cv2.IMREAD_GRAYSCALE)  # queryImage
    #img2 = cv2.imread('Images/right.png', cv2.IMREAD_GRAYSCALE)  # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img3 = resize(img3, 1400, 500)
    return img3

def stitch(img_, img):
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

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
    dst = resize(dst, 1400, 500)
    return dst


def gui():
    sg.theme("LightGreen")

    # Define the window layout

    layout = [

        [   # Added text UI to show some title text
            sg.Text("COM31006: Salient feature, Matching and Image Stitch: Assignment", size=(60, 1), justification="center")
        ],

        [
            sg.Text("Image File 1"),
            sg.In(size=(25, 1), enable_events=True, key="-FILE-"),
            sg.FileBrowse(),
            sg.Text("Image File 2"),
            sg.In(size=(25, 1), enable_events=True, key="-FILE2-"),
            sg.FileBrowse(),
        ],

        [
            sg.Image(filename="", key="-IMAGE-"),
            sg.Image(filename="", key="-IMAGE2-")
        ],

        [sg.Radio("None", "Radio", True, size=(10, 1), key="-NORMAL-")],

        [
            sg.Radio("Harris Corner Detector", "Radio", size=(20, 1), key="-HARRIS CORNER-"),
            sg.Radio("Harris Corner Detector (dilated)", "Radio", size=(30, 1), key="-HARRIS CORNER DILATED-")
        ],

        [
            sg.Radio("SIFT feature point Detector", "Radio", size=(20, 1), key="-SIFT-"),
            sg.Radio("Display flags", "Radio", size=(10, 1), key="-FLAG-"),
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

    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-FILE-":
            filename1 = values ["-FILE-"]
        elif event == "-FILE2-":
            filename2 = values["-FILE2-"]

        if filename1 != "" and filename2 != "":
            frame1 = cv2.imread(filename1)
            frame2 = cv2.imread(filename2)

            if values["-NORMAL-"]:
                show_both = True

                frame1 = cv2.imread(filename1)
                frame1 = resize(frame1, 700, 500)
                frame2 = cv2.imread(filename2)
                frame2 = resize(frame2, 700, 500)

                just_switched = True

            elif values["-HARRIS CORNER-"]:
                show_both = True

                frame1 = harris_corner(frame1)
                frame2 = harris_corner(frame2)

                just_switched = True

            elif values["-HARRIS CORNER DILATED-"]:
                show_both = True

                frame1 = harris_corner_dilated(frame1)
                frame2 = harris_corner_dilated(frame2)

                just_switched = True

            elif values["-SIFT-"]:
                show_both = True

                frame1, kp, des = sift_detector(frame1, False)
                frame2, kp, des = sift_detector(frame2, False)

                just_switched = True

            elif values["-FLAG-"]:
                show_both = True

                frame1, kp, des = sift_detector(frame1, True)
                frame2, kp, des = sift_detector(frame2, True)

                just_switched = True

            elif values["-MATCHER-"]:
                show_both = False

                frame1 = brute_force_matching(frame1, frame2)

                just_switched = True

            elif values["-STITCH-"]:
                show_both = False

                frame1 = stitch(frame2, frame1)

                just_switched = True

        if just_switched and show_both:

            imgbytes = cv2.imencode(".png", frame1)[1].tobytes()
            imgbytes2 = cv2.imencode(".png", frame2)[1].tobytes()

            window["-IMAGE-"].update(data=imgbytes)
            window["-IMAGE2-"].update(data=imgbytes2, visible=True)

            just_switched = False

        elif just_switched and not show_both:

            imgbytes = cv2.imencode(".png", frame1)[1].tobytes()

            window["-IMAGE-"].update(data=imgbytes)
            window["-IMAGE2-"].update(visible=False)

            just_switched = False

    window.close()

if __name__ == '__main__':
    # harris_corner_detector("wall.png")
    # sift_detector("wall.png")
    gui()
    #img = cv2.imread("Images/left.png")
    #img2 = cv2.imread("Images/right.png")
    #stitch(img2, img)