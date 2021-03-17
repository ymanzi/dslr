import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Komparator:
	def __init__(self, df: pd.DataFrame):
		self.data = df

	def compare_box_plots(self, categorical_var, numerical_var):
		lst_categorical = self.data[categorical_var].unique()
		row = int(len(lst_categorical))
		fig, axs = plt.subplots(nrows= 1, ncols= row)
		for i, elem in enumerate(lst_categorical):
			df = self.data[(self.data[categorical_var] == elem)][numerical_var].dropna()
			axs[i].boxplot(df, vert=False, notch = True, labels=list(lst_categorical[i]), whis=0.75, widths = 0.1, patch_artist=bool(i % 2))
			axs[i].set_xlabel(numerical_var)
			axs[i].legend(lst_categorical[i])
		plt.show()

	def density(self, categorical_var, numerical_var) :
		lst_categorical = self.data[categorical_var].unique()
		for elem in lst_categorical:
			data = self.data[(self.data[categorical_var] == elem)][numerical_var]
			sns.distplot(data, hist=False, kde=True, kde_kws={'linewidth': 3}, label = elem)
		plt.legend(prop={'size': 16}, title=categorical_var)
		plt.show()

	def pairplot_(self, categorical_var, numerical_var):
		lst_categorical = self.data[categorical_var].unique()
		my = []
		for elem in lst_categorical:
			data = self.data[numerical_var[0:3]]
			sns.pairplot(data, hue=categorical_var) # HHHHHHHHHEEEEEERRRRRREEEEE
			plt.legend(prop={'size': 16}, title=elem)
			plt.show()
			return

	def compare_histograms(self, categorical_var, numerical_var):
		lst_categorical = self.data[categorical_var].unique()
		row = int(len(numerical_var))
		fig, axis = plt.subplots(nrows= 2, ncols=  row // 2)
		for i in range(0, len(numerical_var)):
			my_list = []
			for j, elem in enumerate(lst_categorical):
				my_list.append (list(self.data[(self.data[categorical_var] == elem)][numerical_var[i]].dropna().values.transpose()))
			axis[i % 2, i // 2].hist(my_list, stacked=False, label=lst_categorical)#, density=True, bins = int(185/15))
			axis[i % 2, i // 2].set_xlabel(numerical_var[i])
			if not(i):
				axis[i % 3, i // 3].legend()
		fig.suptitle("Histograms")
		plt.show()

# from FileLoader import FileLoader
# loader = FileLoader()
# data = loader.load("../resources/athlete_events.csv")
# tmp = Komparator(data)
# tmp.compare_histograms("Sex", "Height")
# tmp.compare_box_plots("Sex", "Height")
# tmp.density("Sex", "Height")