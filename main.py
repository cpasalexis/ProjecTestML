import argparse 
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor

parser = argparse.ArgumentParser(prog = "24/25 ML Project Example",
                                 description = "Example program for the ML project course")

parser.add_argument("--dataset_path",type = str,default = "",help = "path to the dataset file")
parser.add_argument("--ml_method",type = str, default = "Linear", help="name of the ML method")
parser.add_argument("--l2_penalty",type = float, default= 1., help ="Strength of the L2 penalty")
parser.add_argument("--cv_nsplits",type = int, default = 5, help = "cross-validation number")
parser.add_argument("--save_dir", type = str, default = "", help ="where to save the model") 

args = parser.parse_args()
