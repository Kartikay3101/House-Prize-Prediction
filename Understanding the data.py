# Import all Important Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing.values())
print(type(housing))
