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
	rm -f __pycache__

predict:
	python3.8 logreg_predict.py resources/x_test.csv

train:
	python3.8 logreg_train.py resources/x_train.csv

histo:
	python3.8 histogram.py resources/dataset_train.csv

pair:
	python3.8 pair_plot.py resources/dataset_train.csv

scatter:
	python3.8 scatter_plot.py resources/dataset_train.csv