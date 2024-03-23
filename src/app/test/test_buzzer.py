import os

from src.core.alarm.alarm import Alarm
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
alarm = Alarm()
alarm.start()
