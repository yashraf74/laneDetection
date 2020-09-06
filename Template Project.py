import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpclrs
from matplotlib import cm
from time import time
import glob
from time import time
import math
from PIL import Image, ImageEnhance
from scipy import interpolate, ndimage
import colorsys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import skvideo
import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML

team_members_names = ['يوسف اشرف عبد المعبود', 'يوستينا ناصر صادق', 'ناتالى فؤاد مسعد', 'ميريت هانى ابراهيم']
team_members_seatnumbers = ['2016170617', '2016170510', '2016170456', '2016170451']



points = []
frames_to_skip = 5
frames_skipped = frames_to_skip


def draw_lines_connected(img, p0, p1, color=None, thickness=8):

    def draw_pixel(pxl_x, pxl_y):
        if thickness == 0:
            return
        pxl_x = np.clip(pxl_x, 0, img.shape[0] - 1)
        pxl_y = np.clip(pxl_y, 0, img.shape[1] - 1)
        img[pxl_x, pxl_y] = color
        for i in range(1, int(thickness / 2) + 1):
            if pxl_y - i >= 0:
                img[pxl_x, pxl_y - i] = color
        for i in range(1, int(thickness / 2) + 1):
            if pxl_y + i < img.shape[1]:
                img[pxl_x, pxl_y + i] = color

    def plot_high(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        d = (2 * dx) - dy
        x = x0
        for y in range(y0, y1):
            draw_pixel(x, y)
            if d > 0:
                x = x + xi
                d = d - (2 * dy)
            d = d + (2 * dx)

    def plot_low(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        d = (2 * dy) - dx
        y = y0
        for x in range(x0, x1):
            draw_pixel(x, y)
            if d > 0:
                y = y + yi
                d = d - (2 * dx)
            d = d + (2 * dy)

    if color is None:
        color = [255, 0, 0]

    m_x0, m_y0 = p0[0], p0[1]
    m_x1, m_y1 = p1[0], p1[1]

    if abs(m_y1 - m_y0) < abs(m_x1 - m_x0):
        if m_x0 > m_x1:
            plot_low(m_x1, m_y1, m_x0, m_y0)
        else:
            plot_low(m_x0, m_y0, m_x1, m_y1)
    else:
        if m_y0 > m_y1:
            plot_high(m_x1, m_y1, m_x0, m_y0)
        else:
            plot_high(m_x0, m_y0, m_x1, m_y1)
    return


def convert_rbg_to_grayscale(img):

    rgb = np.array(img)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def convert_rgb_to_hsv(img):
    pixels = img.load()
    hls_array = np.empty(shape=(img.height, img.width, 3), dtype=float)
    new_image = Image.new("RGB", (hls_array.shape[1], hls_array.shape[0]))

    for row in range(0, img.height):
        for column in range(0, img.width):
            rgb = pixels[column, row]
            hls = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
            new_image.putpixel((column, row), hls)

    return new_image


def conv(image, kernel):
    image = np.array(image)
    kernel_size = kernel.shape[0]
    pad_w = int(kernel_size / 2)
    img = np.pad(image, [(pad_w, pad_w), (pad_w, pad_w)], 'constant')
    for ii in range(image.shape[0]):
        i = ii + pad_w
        for jj in range(image.shape[1]):
            j = jj + pad_w
            img_kernel = img[i - pad_w:i + pad_w + 1, j - pad_w:j + pad_w + 1]
            image[ii, jj] = np.sum(np.multiply(kernel, img_kernel))
    return Image.fromarray(image)


def sobel_edge(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    img_x = ndimage.filters.convolve(image, kernel_x)
    img_y = ndimage.filters.convolve(image, kernel_y)

    return img_x, img_y


def laplacian_edge(image):
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], np.float32)
    new_img = ndimage.filters.convolve(np.array(image), kernel)

    return Image.fromarray(new_img)


def detect_edges_canny(image, low_thresh_ratio, high_thresh_ratio):
    image = np.array(image)

    img_x, img_y = sobel_edge(image)
    gradient = np.hypot(img_x, img_y)
    gradient = gradient / gradient.max() * 255
    theta = np.arctan2(img_x, img_y)

    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    width, height = gradient.shape
    non_max_supp_img = np.zeros((width, height), dtype=np.int32)
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient[i, j + 1]
                    r = gradient[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient[i + 1, j - 1]
                    r = gradient[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient[i + 1, j]
                    r = gradient[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient[i - 1, j - 1]
                    r = gradient[i + 1, j + 1]

                if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                    non_max_supp_img[i, j] = gradient[i, j]
                else:
                    non_max_supp_img[i, j] = 0

            except IndexError as e:
                pass

    high_threshold = non_max_supp_img.max() * high_thresh_ratio
    low_threshold = high_threshold * low_thresh_ratio

    res = np.zeros((width, height), dtype=np.int32)

    weak_pxl = np.int32(25)
    strong_pxl = np.int32(255)

    strong_i, strong_j = np.where(non_max_supp_img >= high_threshold)
    weak_i, weak_j = np.where((non_max_supp_img <= high_threshold) & (non_max_supp_img >= low_threshold))

    res[strong_i, strong_j] = strong_pxl
    res[weak_i, weak_j] = weak_pxl

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if res[i, j] == weak_pxl:
                try:
                    if ((res[i + 1, j - 1] == strong_pxl) or (res[i + 1, j] == strong_pxl) or (
                            res[i + 1, j + 1] == strong_pxl)
                            or (res[i, j - 1] == strong_pxl) or (res[i, j + 1] == strong_pxl)
                            or (res[i - 1, j - 1] == strong_pxl) or (res[i - 1, j] == strong_pxl)
                            or (res[i - 1, j + 1] == strong_pxl)):
                        res[i, j] = strong_pxl
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass

    return res


def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal

    return g


def remove_noise(img, kernel_size):

    kernel = gaussian_kernel(kernel_size, sigma=kernel_size / 6)
    img = ndimage.filters.convolve(img, kernel)

    return Image.fromarray(img)


def mask_image(rgb_img):
    img = np.array(rgb_img)
    white_low = [200, 200, 200]
    yellow_low = [190, 190, 30]
    mask = np.ones(shape=(img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] < yellow_low[0] or img[i, j, 1] < yellow_low[1] or img[i, j, 2] < yellow_low[2]:
                mask[i, j] = 0
    return np.uint8(mask)


def apply_mask(img_grey, mask):
    img_grey = np.array(img_grey)
    mask = np.array(mask)
    for i in range(img_grey.shape[0]):
        for j in range(img_grey.shape[1]):
            if mask[i, j] == 0:
                img_grey[i, j] = 0
    return Image.fromarray(img_grey)


def read_image(img_path):
    return Image.open(img_path)


def hough_line(edge):
    theta = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))

    rho_range = round(math.sqrt(edge.shape[0] ** 2 + edge.shape[1] ** 2))
    accumulator = np.zeros((2 * rho_range, len(theta)), dtype=np.uint8)

    edge_pixels = np.where(edge == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    for p in range(len(coordinates)):
        for t in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 2

    return accumulator


def visualize_lines(image, lines):
    # print(len(lines))
    try:
        assert len(lines) % 2 == 0 and len(lines) != 0
    except AssertionError:
        lines.pop()
    new_img = np.array(image)
    color = [255, 0, 0]

    ctr = 0
    for i in range(int(len(lines)/2)):
        p1, p2 = lines[ctr], lines[ctr + 1]
        ctr += 2
        x1, y1 = int(p1[0]), p1[1]
        x2, y2 = int(p2[0]), p2[1]
        print(x1, y1)
        print(x2, y2)
        draw_lines_connected(new_img, (y1, x1), (y2, x2), color, 6)

    return new_img


def get_points_from_hough(accum):
    pts = []

    edge_pixels = np.where(accum > 100)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))
    polygon = Polygon([(540, 110),
                       (315, 390),
                       (323, 575),
                       (540, 903)])

    slope = np.inf
    for i in range(0, len(coordinates)):
        a = np.cos(np.deg2rad(coordinates[i][1]))
        b = np.sin(np.deg2rad(coordinates[i][1]))
        x0 = a * coordinates[i][0]
        y0 = b * coordinates[i][0]
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * a)
        curr_slope = (y2 - y1) / (x2 - x1)
        if np.abs(curr_slope - slope) < 0.5:
            slope = curr_slope
            continue
        slope = curr_slope

        x3 = (323 - y2) / ((y1 - y2) / (x1 - x2)) + x2
        y3 = 323
        if y2 > 323:
            x1 = x3
            y1 = y3
        else:
            x2 = x3
            y2 = y3
        # if not polygon.contains(Point(int(x1), y1)) and not polygon.contains(Point(int(x2), y2)):
        #     continue
        pts.append([int(x1), y1])
        pts.append([int(x2), y2])

    return pts


def process_frame(frame):
    global points, frames_skipped, frames_to_skip

    if frames_skipped != frames_to_skip:
        frames_skipped -= 1
        if frames_skipped == 0:
            frames_skipped = frames_to_skip
        return visualize_lines(frame, points)

    tik = time()
    frames_skipped -= 1
    points = []

    gray_img = convert_rbg_to_grayscale(frame)
    mask = mask_image(frame)
    masked_img = apply_mask(gray_img, mask)

    filtered_img = remove_noise(masked_img, kernel_size=19)
    # filtered_img.show()
    edges_img = detect_edges_canny(filtered_img, 0.05, 0.09)

    accumulator = hough_line(edges_img)
    points = get_points_from_hough(accumulator)
    out_img = visualize_lines(frame, points)

    tok = time()
    print("Frame processed in: ", tok - tik, "s")
    # Image.fromarray(out_img).show()

    return out_img


def main():
     input_vid_path_yellow = '/Users/Joe/Desktop/FCIS/Computer vision/Project 1/Input videos/Yello Lane.mp4'
     output_vid_path_yellow = '/Users/Joe/Desktop/FCIS/Computer vision/Project 1/Output videos/Yellow_vid_out.mp4'
    
     input_vid_path_white = '/Users/Joe/Desktop/FCIS/Computer vision/Project 1/Input videos/White Lane.mp4'
     output_vid_path_white = '/Users/Joe/Desktop/FCIS/Computer vision/Project 1/Output videos/White_vid_out.mp4'
    
     tik = time()
     input_video_yellow = VideoFileClip(input_vid_path_yellow, audio=False)
     processed = input_video_yellow.fl_image(process_frame)
     processed.write_videofile(output_vid_path_yellow, audio=False)
     tok = time()
     print("Elapsed time: ", (tok - tik)/60.0, "m")
    
     tik = time()
     input_video_white = VideoFileClip(input_vid_path_white, audio=False)
     processed = input_video_white.fl_image(process_frame)
     processed.write_videofile(output_vid_path_white, audio=False)
     tok = time()
     print("Elapsed time: ", (tok - tik)/60.0, "m")


if __name__ == '__main__':
    main()
