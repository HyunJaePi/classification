# Testing different classifiers

(COSI 123A) Statistical Machine Learning </br>
 
### Task: 1) testing different classifiers on DNA sequence data and 2) get the best prediction accuracy

SVM, Decision Trees, AdaBoost and Bagging were tested. All classifiers gave reasonably good accuracy (around
0.9) without fine tuning of parameters. However, Decision Trees was better than SVM both in single simulations or
with ensemble approaches. After parameter tuning, the accuracies of AdaBoost with Decision Trees and Bagging
with Decision Trees were 0.931 and 0.942, respectively. Therefore, Bagging with Decision Trees was chosen because
it gave the best accuracy 0.942 (cross validation = 5, 31 base learners). The following table summarizes the simulations 
that I run with different classifiers and best parameters.



| Classifier/kernel | Accuracy | Parameter1  |  Parameter2 |
| ------------- |:-------------:| -----:|-----:|
|SVM / rbf  |   0.905  |   C = 1 |    gamma = 0.01 |
|SVM / polynomial | 0.886 | C = 0.001 | gamma = 1 |
|SVM / linear | 0.897 |C = 10 | N/A |
|Decision Trees | 0.924 | max depth = 8 | min samples split = 0.01 |
|AdaBoost w/ Decision Trees | 0.927 | n estimators = 35 | learning rate = 0.6 |
|AdaBoost w/ SVM | 0.931 | n estimators = 40 | learning rate = 0.9 |
|Bagging w/ Decision Trees | 0.942 | n estimators = 31 | N/A |
|Bagging w/ SVM | 0.898 | n estimators = 33 | N/A |
