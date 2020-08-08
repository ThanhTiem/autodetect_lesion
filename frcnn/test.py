from pathlib import Path
import os
import pickle

a = os.path.join(Path(__file__).absolute().parent, "final_config.pickle")
b = pickle.load(open(a, "rb"))
