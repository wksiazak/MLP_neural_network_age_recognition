## ... not finished... Age prediction in MLP neural network project 
This code allows to predict age from photo based on MLP model trained on data available on kaggle dataset [Kaggle age prediction dataset] (https://www.kaggle.com/datasets/mariafrenti/age-prediction/data)

### Table of content 
* [General info](#general-info)
* [Technologies/Libraries](#technologies/libraries)
* [Results](#results)
* [Summary](#summary)

## General info <a name="general-info"></a>

<details>
<summary>Click here to see general information about Age prediction in MLP neural network project!</summary>
<p>Main purpose of this project was to train an MLP model based on available training dataset which could predict the age of people in photos in the testing dataset. 
The project consists of several steps, including:</p>

* Splitting the training dataset into training and validation sets.
* Creating a generator in a separate class to improve model training based on images.
* Building and training the model.
* Evaluating the model.
</details>

## Technolgies <a name="technologies/libraries"></a>
<ul>
<li>Python -  scripts are written in Python</li>
<li>tensorflow</li>
<li>sklearn</li>
<li>matplotlib</li>
</ul>


## Results <a name="results"></a>
<details>
<summary>Click here to see more about <b>details</b>!</summary>
  
Model which was built from seven layers (first flatten, five dense and one output) in 50 epochs shows below losses and MAEs during training: 
![Opis obrazka](https://github.com/wksiazak/MLP_neural_network_age_recognition/issues/1#issue-2614840799)

# Loss over Epochs
- Trend: Both training and validation loss start very high and decrease rapidly within the first 10 epochs. After that, they stabilize and maintain relatively - steady values around 80.
- Stability: This suggests that the model has converged to a point where additional training no longer significantly reduces loss.
- Training vs. Validation: The training and validation losses remain very close to each other, which is a positive sign. If the validation loss were significantly higher than the training loss, it would indicate overfitting, but that doesn’t appear to be the case here.

# Mean Absolute Error (MAE) over Epochs:
- Initial Decrease: Both the training and validation MAE start at high values (~12) and drop sharply in the first 10 epochs, indicating that the model is quickly learning to reduce prediction error.
- Convergence: The MAE stabilizes around 7-8 for both training and validation. This is consistent with your project’s results, as an MAE of 7-8 means that the model’s age predictions are, on average, within 7-8 years of the actual age.
- Consistency: The closeness between training and validation MAE further confirms that the model generalizes well, without substantial overfitting
</details>


## Summary <a name="summary"></a>
<details>

</details>
