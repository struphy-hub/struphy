import os
import logging

# Setup logger.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs('logs', exist_ok=True)
handler = logging.FileHandler('logs/execution.log', 'a', 'utf-8')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Disable numba debug logs.
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Disable matplotlib debug logs.
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
