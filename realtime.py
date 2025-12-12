# realtime.py - BEST APPROACH
import sys
import time
from tqdm import tqdm

# Method 1: Using tqdm (RECOMMENDED)
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.1)
 