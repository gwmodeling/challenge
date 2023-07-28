# Team Example

In this file the submission is described. 

## Author(s)

- Amir Rouhani (1. Department of Aquatic Ecosystem Analysis and Management, Helmholtz Centre for Environmental Research - UFZ, Magdeburg, Germany. 
	2. Department of Environment, Land and Infrastructure Engineering, Politecnico di Torino, Turin, Italy. (1. amir.rouhani@ufz.de, 2. amir.rouhani@studenti.polito.it ))
- J. Jaime Gómez-Hernández (Institute for Water and Environmental Engineering, Universitat Politècnica de València, Valencia, Spain, (jgomez@upv.es ))
- Seifeddine Jomaa (Department of Aquatic Ecosystem Analysis and Management, Helmholtz Centre for Environmental Research - UFZ, Magdeburg, Germany (seifeddine.jomaa@ufz.de ))

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [x] USA

## Feedback / Ideas for model evaluation

- Considering different statistical metrics (NSE, RMSE/rRMSE, R2, Bias/rBias), including the extreme groundwater levels 
  (highs/lows) and groundwater balance, is recommended for model evaluation as each metric gives a particular assessment for a specific groundwater pattern, 
- The NSE is an appropriate statistical metric to evaluate better extreme groundwater levels, 
- Combining multiple statistical metrics with visual evaluation can be a better approach for model results assessment.   

## Model description

We used Convolutional Neural Networks (CNNs) model developed by Wunsch et al. (2022) [1]. The CNNs model utilized in this work includes a 1-D convolutional layer with a fixed kernel 
size (three) and an optimal number of filters, followed by a Max-Pooling layer and a Monte-Carlo dropout layer with a fixed dropout of 50% to prevent overfitting. This high dropout 
rate necessitates solid training for the model. Following that is a thick layer with an optimal number of neurons, followed by a single output neuron. The Adam optimizer was used for 
a maximum of 100 training epochs with an initial learning rate of 0.001, and gradient clipping was utilized to prevent exploding gradients. Another regularization strategy that was 
considered to prevent the model from overfitting the training data was early halting with a patience of 15 epochs. Bayesian optimization was used to tune several model hyperparameters (HP) [2]: 
training batch size (16-256); input sequence length (1-365 for daily datasets) & (1-52 for weekly datasets); the number of filters in the 1D-Conv layer (1-256); and the size of the first dense 
layer (1-256). All models were built with Python 3.8 [3], the TensorFlow deep-learning framework [4], and its Keras API [5]. NumPy [6], Pandas [7], [8], Scikit-Learn [9], BayesOpt [10], Matplotlib [11], 
UnumPy [12] libraries were also utilized. [1].

## Model workflow to reproduce

We use the parameters provided for each time series data to train the model and divide each time series into four parts to identify the optimum model configuration: training set, validation set, 
optimization set, and test set. The test set always uses the most recent four years of data provided. The first 80% of the remaining time series were utilized for training, the next 20% for early 
stopping (validation set), and the remaining 10% for testing during HP tuning (optimization set), each using 10% of the remaining time series. We used a maximum optimization step number of 150 for 
each model or stopped after 15 steps without improvement if a minimum of 60 steps was reached. To lessen reliance on the random number generator seed, we scaled the data to [-1,1] and employed an 
ensemble of 10 pseudo-randomly started models. We used Monte-Carlo dropout during simulation to estimate model uncertainty from 100 realizations for each of the ten ensemble members. Using 1.96 times 
the standard deviation of the resulting distribution for each time step, we calculated the 95% confidence interval from these 100 realizations. To assess simulation accuracy, we measured NSE, 
squared Pearson r (R2), absolute and relative root mean squared error (RMSE/rRMSE), and absolute and relative Bias (Bias/rBias). We calculate NSE using a long-term mean of groundwater level before 
the test set rather than the test set mean value [13].

## References 

[1]	A. Wunsch, T. Liesch, and S. Broda, “Deep learning shows declining groundwater levels in Germany until 2100 due to climate change,” Nature Communications 2022 13:1, vol. 13, no. 1, pp. 1–13, Mar. 2022, doi: 10.1038/s41467-022-28770-2.
[2]	F Nogueira, “Bayesian Optimization: Open source constrained global optimization tool for Python,” 2014. https://github.com/fmfn/BayesianOptimization (accessed Jan. 12, 2023).
[3]	“Python Tutorial Release 3.8.1 Guido van Rossum and the Python development team,” 2020.
[4]	M. Abadi et al., “TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems,” Mar. 2016, doi: 10.48550/arxiv.1603.04467.
[5]	F. Chollet, “keras,” 2015. https://github.com/fchollet/keras (accessed Jan. 12, 2023).
[6]	S. van der Walt, S. C. Colbert, and G. Varoquaux, “The NumPy array: A structure for efficient numerical computation,” Comput Sci Eng, vol. 13, no. 2, pp. 22–30, Mar. 2011, doi: 10.1109/MCSE.2011.37.
[7]	W. Mckinney, “Data Structures for Statistical Computing in Python,” 2010.
[8]	T. pandas development team, “pandas-dev/pandas: Pandas,” Nov. 2022, doi: 10.5281/ZENODO.7344967.
[9]	F. Pedregosa FABIANPEDREGOSA et al., “Scikit-learn: Machine Learning in Python Gaël Varoquaux Bertrand Thirion Vincent Dubourg Alexandre Passos PEDREGOSA, VAROQUAUX, GRAMFORT ET AL. Matthieu Perrot,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011, Accessed: Jan. 12, 2023. [Online]. Available: http://scikit-learn.sourceforge.net.
[10]	“fmfn/BayesianOptimization: A Python implementation of global optimization with gaussian processes.” https://github.com/fmfn/BayesianOptimization (accessed Jan. 12, 2023).
[11]	J. D. Hunter, “Matplotlib: A 2D Graphics Environment,” Comput Sci Eng, vol. 9, no. 03, pp. 90–95, May 2007, doi: 10.1109/MCSE.2007.55.
[12]	“Welcome to the uncertainties package — uncertainties Python package 3.0.1 documentation.” https://pythonhosted.org/uncertainties/ (accessed Jan. 12, 2023).
[13]	A. Wunsch, T. Liesch, and S. Broda, “Groundwater level forecasting with artificial neural networks: A comparison of long short-term memory (LSTM), convolutional neural networks (CNNs), and non-linear autoregressive networks with exogenous input (NARX),” Hydrol Earth Syst Sci, vol. 25, no. 3, pp. 1671–1687, Apr. 2021, doi: 10.5194/HESS-25-1671-2021.


## Supplementary model data used

No additional information


## Estimation of effort

Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz   2.00 GHz processor
8.00 GB RAM 

| Location    |   Training time (hrs)  | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~ 3                    |                      | 03:00:00         |
| Germany     | ~ 3                    |                      | 03:00:00         |
| Sweden 1    | ~ 1                    |                      | 01:00:00         |
| Sweden 2    | ~ 1                    |                      | 01:00:00         |
| USA         | ~ 3                    |                      | 03:00:00         |

## Additional information

If you want to provide any additional information about your model/submission/results, please do so here.