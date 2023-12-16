import os
from src.core import TOMLConfig, Alarm

setting = TOMLConfig(os.path.join(__file__, "../config.toml"))
alarm = Alarm()
alarm.start()
