import os, re, cv2
import numpy as np
import matplotlib.pyplot as plt


def get_sorted_corner(corners):
    """
    Sort the corners for calibration --> To establish correspondence as both calibration
    and distorted image must follow the sorting algorithm

    :param corners  :   The detected corners (list)
    :return         :   A sorted list of corners (list)
    """
    sorted_x_corners = sorted(corners, key=lambda p: p[0])
    sorted_corners = []

    index = 0
    cur_x_val = sorted_x_corners[0][0]
    for i in range(len(sorted_x_corners)):
        x, _ = sorted_x_corners[i]
        if np.abs(x - cur_x_val) >= 5:
            if x != cur_x_val:
                sorted_corners.extend(sorted(sorted_x_corners[index:i], key=lambda p: p[1]))
                index = i
                cur_x_val = x

    sorted_corners.extend(sorted(sorted_x_corners[index:], key=lambda p: p[1]))

    return sorted_corners


def manual_detection_corners(img, detected_corners, handle_mouse_bool=False):
    """
    The detection algorithm using Harris corner might not capture all the corners of the
    checkerboard. Allow user to input and delete points.

    :param img                  :   Image of the checkerboard
    :param detected_corners     :   The detected corners using the Harris corner algorithm
    :param handle_mouse_bool    :   Whether user input is needed (True if yes else no)
    :return                     :   A new list of corners with the added/deleted corners
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = img.copy()
    for corner in detected_corners:
        x, y = np.intp(corner)
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Corners', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Corners', img_copy)

    # Sub_Pixel Accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if handle_mouse_bool:
        def handle_mouse(event, x, y, flags, param):
            nonlocal img_copy, detected_corners

            if event == cv2.EVENT_LBUTTONDOWN:
                corners = np.array([[[x, y]]], dtype=np.float32)
                refined_point = cv2.cornerSubPix(gray, corners, winSize=(5, 5), zeroZone=(-1, -1),
                                                 criteria=criteria)
                refined_point = refined_point[0, 0]
                detected_corners = np.vstack([detected_corners, refined_point])

                cv2.circle(img_copy, np.intp(refined_point), 5, (0, 0, 255), -1)
                cv2.imshow('Corners', img_copy)

            elif event == cv2.EVENT_RBUTTONDOWN:
                for i, corner in enumerate(detected_corners):
                    dist = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
                    if dist <= 5:
                        detected_corners = np.delete(detected_corners, i, axis=0)
                        img_copy = img.copy()
                        for corner in detected_corners:
                            cv2.circle(img_copy, np.intp(corner), 5, (0, 255, 0), -1)
                        cv2.imshow('Corners', img_copy)

        cv2.setMouseCallback('Corners', handle_mouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detected_corners


def detect_corners(image_set_path):
    """
    Detect the corner to subpixel accuracy with the Harris Corner Detection algorithm provided
    by OpenCV

    :param image_set_path   :   Path to input image set
    :return                 :   A list of corners of the checkerboard
    """
    assert isinstance(image_set_path, np.ndarray), "Incorrect array structure: requires Numpy array."

    all_img_corners = []
    for image in image_set_path:
        img = cv2.imread(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        _, _, _, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids[1:]), (11, 11), (-1, -1), criteria)
        single_img_corners = manual_detection_corners(img, corners, handle_mouse_bool=True)
        sorted_corners = np.array(get_sorted_corner(single_img_corners))

        i = 0
        while i < len(sorted_corners) - 1:
            if np.sqrt((sorted_corners[i, 0] - sorted_corners[i + 1, 0]) ** 2 +
                       (sorted_corners[i, 1] - sorted_corners[i + 1, 1]) ** 2) < 1:
                sorted_corners = np.delete(sorted_corners, i + 1, axis=0)
            else:
                i += 1
        all_img_corners.extend(sorted_corners)
    all_img_corners = np.array(all_img_corners, dtype=object)
    return all_img_corners


if __name__ == "__main__":

    ref = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\out_of_plane_sensitivity\ref.png"
    tar_0 = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\out_of_plane_sensitivity\p_0.png"
    tar_10 = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\out_of_plane_sensitivity\p_0.01.png"
    tar_20 = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\out_of_plane_sensitivity\p_0.02.png"
    tar_50 = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\out_of_plane_sensitivity\p_0.05.png"
    save_path = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\out_of_plane_sensitivity"

    ref_pts = detect_corners(np.array([ref]))
    base_vel = ref_pts - ref_pts
    tar_0_pts = detect_corners(np.array([tar_0]))
    tar_10_pts = detect_corners(np.array([tar_10]))
    tar_20_pts = detect_corners(np.array([tar_20]))
    tar_50_pts = detect_corners(np.array([tar_50]))

    vel_0 = tar_0_pts - ref_pts
    vel_10 = tar_10_pts - ref_pts
    vel_20 = tar_20_pts - ref_pts
    vel_50 = tar_50_pts - ref_pts
    
    for label, vel in [('p=0.00', vel_0), ('p=0.01', vel_10), ('p=0.02', vel_20), ('p=0.05', vel_50)]:
        vx, vy = vel[:, 0].astype(float), vel[:, 1].astype(float)
        print(f"{label}  |  vx: mean={vx.mean():.4f}, std={vx.std():.4f}  |  vy: mean={vy.mean():.4f}, std={vy.std():.4f}")

    labels = ['0 mm', '10 mm', '20 mm', '50 mm']
    out_of_plane_mm = [0, 10, 20, 50]
    cases = [vel_0, vel_10, vel_20, vel_50]

    # Subtract ground truth (vel_0) to isolate out-of-plane-induced error
    gt_vx = vel_0[:, 0].astype(float)
    gt_vy = vel_0[:, 1].astype(float)

    means_vx, stds_vx, means_vy, stds_vy = [], [], [], []
    for vel in cases:
        vx = vel[:, 0].astype(float) - gt_vx
        vy = vel[:, 1].astype(float) - gt_vy
        means_vx.append(vx.mean())
        stds_vx.append(vx.std())
        means_vy.append(vy.mean())
        stds_vy.append(vy.std())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, means, stds, axis_label in [
        (axes[0], means_vx, stds_vx, 'X'),
        (axes[1], means_vy, stds_vy, 'Y'),
    ]:
        ax.errorbar(out_of_plane_mm, means, yerr=stds,
                    fmt='o-', capsize=6, capthick=1.5, linewidth=1.5, markersize=6, color='black')
        ax.fill_between(out_of_plane_mm,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color='gray', alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f'{axis_label}-axis', fontsize=11)
        ax.set_xticks(out_of_plane_mm)
        ax.set_xticklabels(labels)

    axes[1].tick_params(labelleft=False)
    fig.supxlabel('Out-of-Plane Displacement (mm)', fontsize=11)
    fig.supylabel('Displacement Error (px)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'out_of_plane_sensitivity.png'), dpi=150)
    plt.show()