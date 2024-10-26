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
![Opis obrazka](https://raw.githubusercontent.com/wksiazak/MLP_neural_network_age_recognition/master/examples.png)

MAE (Mean Absolute error) on average is between 7 and 8 which means for our dataset consisting of people photos that predicted age may be even with 7 years error.
In attached examples.png we can see that how trained model coped with testing photos.  
  
</details>


## Summary <a name="summary"></a>
<details>

</details>
