# Multitask Multilabel Wellbeing Prediction

The MTML-NN.py file is for the doctor/nurses job-role based multitask multilabel model proposed in the paper:

Han Yu, Akane Sano, "Forecasting Health and Wellbeing for Shift Workers Using Job-role Based Deep Neural Network", EAI MobiHealth 2020 - 9th EAI International Conference on Wireless Mobile Communication and Healthcare

Enviornment:
- Python version: 3.6
- Python libraries:
--numpy,
--pandas,
--sklearn,
--tensorflow-gpu: 1.12.0
--keras,
--pickle

This implementation includes the models for 3 different tasks -- binary classification, 3-class classification, and regression. To use MTML_NN model, you can import our MTML_NN model and initialize it by for example:

<code>model = MTML_NN(featureDimension = X_train.shape[1], task_type = 'reg')</code>

Note that the last column if the features input should be the cohorts of samples, e.g. nurses(N) or doctors(D).

Then, you can train and validate the model by

<code>model.train(X_train, y_train)</code>

<code>model.predict(X_test)</code>
