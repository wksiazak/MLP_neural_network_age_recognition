## Age prediction in MLP neural network project 
This code allows to predict age from photo based on MLP model trained on data available on [kaggle dataset](https://www.kaggle.com/datasets/mariafrenti/age-prediction/data)

## General info <a name="general-info"></a>

<p>Main purpose of this project was to train an MLP model based on available training dataset which could predict the age of people in photos in the testing dataset. Dataset consists of photos in age between 20 and 50. 
The project consists of several steps, including:</p>

* Splitting the training dataset into training and validation sets.
* Creating a generator in a separate class to improve model training based on images.
* Building and training the model.
* Evaluating the model.


## Technolgies <a name="technologies/libraries"></a>
<ul>
<li>Python -  scripts are written in Python</li>
<li>tensorflow</li>
<li>sklearn</li>
<li>matplotlib</li>
</ul>


## Results <a name="results"></a>

Model which was built from seven layers (first flatten, five dense and one output) in 50 epochs shows below losses and MAEs during training: 

![Loss & MAE](https://github.com/wksiazak/MLP_neural_network_age_recognition/blob/master/final_figure.png)

### Loss over Epochs
- Trend: Both training and validation loss start very high and decrease rapidly within the first 10 epochs. After that, they stabilize and maintain relatively - steady values around 80.
- Stability: This suggests that the model has converged to a point where additional training no longer significantly reduces loss.
- Training vs. Validation: The training and validation losses remain very close to each other, which is a positive sign. If the validation loss were significantly higher than the training loss, it would indicate overfitting, but that doesn’t appear to be the case here.

### Mean Absolute Error (MAE) over Epochs:
- Initial Decrease: Both the training and validation MAE start at high values (~12) and drop sharply in the first 10 epochs, indicating that the model is quickly learning to reduce prediction error.
- Convergence: The MAE stabilizes around 7-8 for both training and validation. This is consistent with your project’s results, as an MAE of 7-8 means that the model’s age predictions are, on average, within 7-8 years of the actual age.
- Consistency: The closeness between training and validation MAE further confirms that the model generalizes well, without substantial overfitting

### Testing photos with predicted age
![Examples](https://github.com/wksiazak/MLP_neural_network_age_recognition/blob/master/examples.png)

In above we can see few examples of testing photos with predicted age (based on trained MLP model)
In those examples it looks like the model frequently predicts ages around 20, regardless of the actual age range of the individual. This indicates that the model may be biased towards a specific age range, possibly due to an imbalance in the training data or limitations in the model architecture's ability to capture more nuanced age-related features. 

## Summary
In general trained model is predicting quite good age on testing dataset, the average difference of around 7 years is acceptable in this dataset.If we could have more data probably our results could be better.  

### Possible Improvements:
 - Data Augmentation: Introducing a more diverse range of ages in the dataset or augmenting images to balance age representation could help reduce age bias.
 - Model Complexity: Using a more complex neural network architecture, such as a Convolutional Neural Network (CNN), could improve feature extraction, especially for distinguishing subtle age differences.
- Additional Regularization: This could help the model generalize better by avoiding over-reliance on any specific facial features.
