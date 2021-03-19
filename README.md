# Datascience X Logistic Regression

**Subject**: [Dslr](https://cdn.intra.42.fr/pdf/pdf/19501/en.subject.pdf)

**Member**: :last_quarter_moon_with_face: [Ymanzi](https://github.com/ymanzi) :first_quarter_moon_with_face:

## How to Use It

### ➣ Make train
**Will train the model with the dataset and generate a file ``theta.csv`` with the weights used for prediction**
``` python3.8 logreg_train.py DATASET_NAME.csv ```

### ➣ Make predict
**Will generate a prediction file *houses.csv* from the dataset and a csv file containing the weights**
``` python3.8 logreg_predict.py DATASET_NAME.csv WEIGHTS.csv```

## Resources

* [Logistic Regression Explained](https://towardsdatascience.com/logistic-regression-explained-9ee73cede081)
* [Logistic Regression Module 8](https://github.com/42-AI/bootcamp_machine-learning)
* [Gradient descent optimization](https://ruder.io/optimizing-gradient-descent/index.html#batchgradientdescent)
* [Stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)
* [Youtube - Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
* [Youtube - Introduction to Logistic Regression](https://www.youtube.com/watch?v=zAULhNrnuL4)

## Visualization

### ➣ Make describe

  ``` python3.8 describe.py DATASET_NAME.csv```
  
**Display information for all numerical *features*.**


<img src="https://github.com/ymanzi/dslr/blob/main/images/describe.png" alt="Describe" width=500 height=250>

### ➣ Make histo

``` python3.8 histogram.py DATASET_NAME.csv ```

**Display a histogram** answering the question:
* Which Hogwarts course has a homogeneous score distribution between all four houses?



<img src="https://github.com/ymanzi/dslr/blob/main/images/histogram.png" alt="Histogram" width=500 height=350>

### ➣ Make pair

``` python3.8 pair_plot.py DATASET_NAME.csv ```

**Display a pair plot** answering the question:
* From this visualization, what features are you going to use for your logistic regression??


<img src="https://github.com/ymanzi/dslr/blob/main/images/pair_plot.png" alt="pair plot" width=500 height=350>

### ➣ Make scatter

``` python3.8 scatter_plot.py DATASET_NAME.csv ```

**Display a scatter plot** answering the question:
* What are the two features that are similar ?

<img src="https://github.com/ymanzi/dslr/blob/main/images/scatter_plot.png" alt="scatter plot" width=500 height=350>
