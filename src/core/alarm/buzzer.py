import RPi.GPIO as GPIO

output_pins = {
    "JETSON_XAVIER": 18,
    "JETSON_NANO": 33,
    "JETSON_NX": 33,
    "CLARA_AGX_XAVIER": 18,
    "JETSON_TX2_NX": 32,
    "JETSON_ORIN": 18,
    "JETSON_ORIN_NX": 33,
    "JETSON_ORIN_NANO": 33,
}


def setup():
    output_pin = output_pins.get(GPIO.model, None)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(output_pin, GPIO.OUT)
    return GPIO.PWM(output_pin, 0)


def cleanup():
    GPIO.cleanup()
