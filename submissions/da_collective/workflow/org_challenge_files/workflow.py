import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def initial_explore():

	inp_name = "input_data.csv"
	h_name = "heads.csv"
	data_dirs = [d for d in os.listdir() if os.path.isdir(d)]
	with PdfPages("data.pdf") as pdf:
		for data_dir in data_dirs:
			idf = pd.read_csv(os.path.join(data_dir,inp_name),index_col=0,parse_dates=[0])
			hdf = pd.read_csv(os.path.join(data_dir,h_name),index_col=0,parse_dates=[0])
			#hdf = hdf.reindex(idf.index)
			#print(idf.columns)
			print(data_dir,hdf.columns,hdf.index,hdf.describe())
			fig,axes = plt.subplots(idf.shape[1]+1,1,figsize=(10,2*idf.shape[1]+1))
			idf.plot(ax=axes[1:],subplots=True)
			axes[0].plot(hdf.index,hdf.values)
			axes[0].set_xlim(axes[1].get_xlim())
			axes[0].set_title(data_dir,loc="left")
			plt.tight_layout()
			pdf.savefig()
			plt.close()


if __name__ == "__main__":
	initial_explore()