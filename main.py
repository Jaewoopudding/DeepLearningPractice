import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

x = torch.ones(1,2,3)
print(x.requires_grad)