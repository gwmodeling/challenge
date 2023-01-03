# Team Selina & Yang

In this file the submission is described. 

## Author(s)

- Xinyue (Selina) Wang (Brown University DSI)
- Yang Zheng (Brown University DSI)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [ ] Sweden 1
- [ ] Sweden 2
- [ ] USA

## Model description


For both locations, we used a Support Vector Machine Regression model (SVR). This is a machine learning model. The model is implemented using the sklearn packaged in Python (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html).



## Model workflow to reproduce

### Overview:
- Using the input data, we created addional features, which include the mean value of each feature over the past 15, 20 and 90 days, as well as a 'day' feature that specifies the day of the month
- We use the MSE as our evaluation metric, and tried three different ML algorithms: Lasso regression, SVR, and random forest regressor
- We split the training data further into training and testing sets, train the model on the training set, predict on the test set, and calculate the MSE score on the test set. We repeat this procedure for 5 iterations, each interation consists of different test set, and a training set that includes all the datapoints that are earlier in time than the test set. This gives 5 different models and 5 different MSE test scores for each ML algorithm.
- In both locations (Germany and Netherlands), the SVR algorithm performed the best, in terms of having a low mean MSE score, low variance in the MSE scores as well as a relatively shorter run time
- We then use these 5 models to produce 5 predictions on the test data used for submission, and take the average of the 5 predictions as our final prediction. We take the square root of the mean of the 5 MSE scores as the RMSE of our prediction. The 95% prediction interval is calculated by adding and subtracting 1.96*RMSE to our final prediction.

### Code:
Please refer to: \
[Germany code](Germany_prediction.ipynb)\
[Netherlands code](Netherlands_prediction.ipynb)

### Python and packaged versions:
Python version 3.10.5\
numpy version 1.22.4\
matplotlib version 3.5.2\
sklearn version 1.1.1\
pandas version 1.4.2


## Supplementary model data used

No additional information was obtained and/or used.

## Estimation of effort

Please provide an (rough) estimate of the time it took to develop the models (e.g., read in data, pick a model 
structure, etcetera) and calibrate the parameters (if any). If possible, please also state the computational resources that 
were required.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 4                    | 140                  | 04:02:20         |
| Germany     | ~ 30                   | 168                  | 30:02:48         |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         |                        |                      |                  |

Most of the time was spent developing the model for Germany, and then we applied the similar procedure to the Netherlands data. The calibration time recorded here is the time it took to train the models.


## Additional information

For the Netherlands submission, we included two files. The [submission form](submission_form_Netherlands.csv) only includes predictions for the days present in the form, and the [full simulation results](full_simulation_results_Netherlands.csv) contains predictions for all the days after 2000-01-01.
