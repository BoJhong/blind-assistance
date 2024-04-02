import cv2
import numpy as np


def detect_blur_fft(image, size=60, thresh=20):
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # 取得圖像的尺寸並推導出中心（x，y）坐標
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean, mean <= thresh


def draw_blur_status(image, mean, blurry):
    image = image.copy()
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image
