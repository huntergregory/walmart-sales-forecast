Tensorflow implementation of a Multi-Horizon Quantile RNN for predicting uncertainty distributions of sales (or other time series data). Based on the [paper](https://arxiv.org/pdf/1711.11053.pdf) by Amazon.

This model was designed specifically for a [Kaggle competition](https://www.kaggle.com/c/m5-forecasting-uncertainty/overview) on predicting 4 week quantile-forecasts for hierarchical Walmart sales data. The data used can be downloaded on the competition page.

To adapt to your own data, update data-generator.py and modify the parameters passed into the data generator in mq-rnn.py.

Future work items include normalizing time series input along each sample plus turning the model in mq-rnn.py into a function for quicker hyperparameter variations and easier setup with different data for other problems.
