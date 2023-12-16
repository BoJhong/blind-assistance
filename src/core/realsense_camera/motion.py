import math

import cv2

first = True
alpha = 0.98
total_gyro_angle_y = -180

accel_angle_x: float
accel_angle_y: float
accel_angle_z: float
last_ts_gyro: float


def get_motion(frames):
    global first, alpha, total_gyro_angle_y, accel_angle_z, accel_angle_x, accel_angle_y, last_ts_gyro
    # gather IMU data
    accel = frames[2].as_motion_frame().get_motion_data()
    gyro = frames[3].as_motion_frame().get_motion_data()

    timestamp = frames.get_timestamp()

    # calculation for the first frame
    if first:
        first = False
        last_ts_gyro = timestamp

        # accelerometer calculation
        accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
        accel_angle_x = math.degrees(
            math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z))
        )
        accel_angle_y = math.degrees(math.pi)

        return

    # calculation for the second frame onwards

    # gyro meter calculations
    dt_gyro = (timestamp - last_ts_gyro) / 1000
    last_ts_gyro = timestamp

    gyro_angle_x = gyro.x * dt_gyro
    gyro_angle_y = gyro.y * dt_gyro
    gyro_angle_z = gyro.z * dt_gyro

    dangleX = gyro_angle_x * 57.2958
    dangleY = gyro_angle_y * 57.2958
    dangleZ = gyro_angle_z * 57.2958

    total_gyro_angle_x = accel_angle_x + dangleX
    # total_gyro_angle_y = accel_angle_y + dangleY
    total_gyro_angle_y = accel_angle_y + dangleY + total_gyro_angle_y
    total_gyro_angle_z = accel_angle_z + dangleZ

    # accelerometer calculation
    accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
    accel_angle_x = math.degrees(
        math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z))
    )
    # accel_angle_y = math.degrees(math.pi)
    accel_angle_y = 0

    # combining gyro meter and accelerometer angles
    combined_angle_x = total_gyro_angle_x * alpha + accel_angle_x * (1 - alpha)
    combined_angle_z = total_gyro_angle_z * alpha + accel_angle_z * (1 - alpha)
    combined_angle_y = total_gyro_angle_y

    pitch = combined_angle_z
    yaw = total_gyro_angle_y
    roll = combined_angle_x

    return pitch, yaw, roll


def draw_motion(image, pitch, yaw, roll):
    image = image.copy()
    cv2.putText(
        image,
        f"pitch: {round(pitch)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"yaw: {round(yaw)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"roll: {round(roll)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    return image
