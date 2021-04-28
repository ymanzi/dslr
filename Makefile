# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ymanzi <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/10/09 14:05:13 by ymanzi            #+#    #+#              #
#    Updated: 2020/11/22 13:49:15 by ymanzi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew
# mkdir ~/.linuxbrew/bin
# ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin
# eval $(~/.linuxbrew/bin/brew shellenv)

env:
	brew install python@3.8
	python -m pip install -U pip
	pip install numpy
	pip install pandas
	pip install matplotlib

clean:
	rm -f __pycache__ houses.csv theta.csv resources/x* resources/y*

describe:
	python3 describe.py resources/dataset_train.csv

predict:
	python3 logreg_predict.py resources/dataset_test.csv theta.csv

train:
	python3 logreg_train.py resources/dataset_train.csv # batch

histo:
	python3 histogram.py resources/dataset_train.csv

pair:
	python3 pair_plot.py resources/dataset_train.csv

scatter:
	python3 scatter_plot.py resources/dataset_train.csv
