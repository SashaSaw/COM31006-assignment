import cv2
import numpy as np
import PySimpleGUI as sg

def resize(img, w, h):
    if img.shape[1] > w:
        img = cv2.resize(img, [w, img.shape[0]])

    if img.shape[0] > h:
        img = cv2.resize(img, [img.shape[1], h])
    return img

def harris_corner(img):
    # Convert cv2 Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate intensity gradients using cv2 Sobel filter
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)


    # calculate products of gradients to get harris matrix
    harris_matrix = [[Ix ** 2, Iy * Ix], [Iy * Ix, Iy ** 2]]
    # Harris matrix:    Ixx | Ixy
    #                   ----|----
    #                   Ixy | Iyy

    # Apply a Gaussian filter to the products of gradients
    # Blur smooths out noise in image & gives more reliable results
    sigma = 1.5 # standard deviation for the gaussian kernal
    harris_matrix[0][0] = cv2.GaussianBlur(harris_matrix[0][0], (3, 3), sigma)
    harris_matrix[1][1] = cv2.GaussianBlur(harris_matrix[1][1], (3, 3), sigma)
    harris_matrix[0][1] = cv2.GaussianBlur(harris_matrix[0][1], (3, 3), sigma)
    harris_matrix[1][0] = cv2.GaussianBlur(harris_matrix[1][0], (3, 3), sigma)

    # Compute Harris corner response
    k = 0.04 # a standard value for k (found in lecture 11)
    # subtract product of top right and bottom left
    # from product of values on the diagonal of the harris matrix
    det = (harris_matrix[0][0] * harris_matrix[1][1]) - (harris_matrix[0][1] * harris_matrix[1][0])
    trace = harris_matrix[0][0] + harris_matrix[1][1] # add values on the diagonal of the harris matrix
    R = det - k * (trace ** 2) # calculate corner score

    # if R is large (greater than 1% of biggest value)
    # then that pixel is a corner - update image to paint pixel red
    img[R > 0.01 * R.max()] = [0, 0, 255]
    # if R is large negative (less than 0.1% of most negative value)
    # then that pixel is an edge - update image to paint pixel blue
    img[R < 0.001 * R.min()] = [255, 0, 0]

    return img

def sift_detector(img, flags, return_image):
    # Convert cv2 Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use cv2 to create a SIFT object
    sift = cv2.SIFT_create()
    # use sift object to detect key points in the grayscale image
    kp = sift.detect(gray, None)

    if flags and return_image: # use key points and the image to draw them onto the image with flags
        new_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif not flags and return_image: # use key points and the image to draw them onto the image without flags
        new_img = cv2.drawKeypoints(img, kp, img)

    # use sift object to compute the descriptors
    kp, des = sift.compute(gray, kp)

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

def stitch(img_left, img_right):

    kp1, des1 = sift_detector(img_right, False, False)
    kp2, des2 = sift_detector(img_left, False, False)

    # find matches using left as query descriptors and right as training descriptors
    matches = two_best_matcher(des1, des2)

    # Apply ratio test
    good = []
    Threshold = 0.7
    for m in matches:
        if m[0].distance / m[1].distance < Threshold:
            good.append([m[0]])
    matches = np.asarray(good)

    # cv.drawMatchesKnn expects list of lists as matches.
    best_matches = matches[:, 0]
    if len(best_matches) >= 4:
        src = []
        dst = []
        for m in best_matches:
            src.append(kp1[m.queryIdx].pt)
            dst.append(kp2[m.trainIdx].pt)
        src = np.float32(src)#.reshape(-1, 1, 2)
        dst = np.float32(dst)#.reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        print("not enough keypoints")

    stitched_image = cv2.warpPerspective(img_right, H, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
    stitched_image[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
    return stitched_image

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
            sg.Radio("SIFT feature point Detector", "Radio", size=(20, 1), key="-SIFT-"),
            sg.Radio("SIFT feature point Detector (Display flags)", "Radio", size=(40, 1), key="-FLAG-"),
        ],

        [sg.Radio("Feature Matcher (SIFT detector, SDD, and Ratio Test)", "Radio", size=(50, 1), key="-MATCHER-")],

        [sg.Radio("Images Stitcher (for SIFT detector, SDD, and Ratio Test Feature Matcher)", "Radio", size=(55, 1), key="-STITCH-")],

        [
            sg.Button("Exit", size=(10, 1)),
            sg.Button("Apply", size=(10, 1))
        ],

    ]
    # Create the window and show it without the plot

    window = sg.Window("COM31006 assignment Alexander Saw", layout, location=(200, 200))

    just_switched = False
    show_both = False
    filename1 = "Images/left.png"
    filename2 = "Images/right.png"
    prev = ""

    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-FILE-":
            filename1 = values ["-FILE-"]
            prev = ""
        elif event == "-FILE2-":
            filename2 = values["-FILE2-"]
            prev = ""
        elif event == "Apply":
            if filename1 != "" and filename2 != "":

                if values["-HARRIS CORNER-"]:
                    show_both = True

                    if prev != "harris corner":
                        frame1, frame2 = setup_frames(filename1, filename2)
                        frame1 = harris_corner(frame1)
                        frame2 = harris_corner(frame2)
                        just_switched = True
                        prev = "harris corner"

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

        if values["-NORMAL-"]:
            show_both = True

            if prev != "normal":
                frame1, frame2 = setup_frames(filename1, filename2)
                just_switched = True
                prev = "normal"

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
    gui()