# Wine Quality - From Theory to Technical Application and Results

This project focuses on exploring and analyzing a dataset of wine quality. The dataset contains information about various quantitative attributes of wine, such as alcohol content, pH level, and more. The quality of each wine was determined by a professional wine sommelier.

The project can be approached in two different ways:

1. Classification Task: Treating 'Quality' as a discrete value, where it can only be a natural number ranging from 1 to 10. Alternatively, the quality can be categorized as either good or bad based on a certain threshold. This approach involves building classification models.

2. Regression Task: Treating 'Quality' as a continuous value, allowing it to be any real number from 1 to 10. This approach involves building regression models.

## Dataset
The dataset used in this project is sourced from a study on wine preferences, specifically modeling wine preferences based on physicochemical properties. The dataset is publicly available and can be accessed from the following source:

- Dataset Link: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## Exploratory Data Analysis
The project begins with exploratory data analysis (EDA) that was performed in jupyter notebook to gain insights and understand the characteristics of the dataset. The EDA is performed separately for red and white wines. The following steps are taken:

1. Loading the dataset: The red wine dataset and the white wine dataset are loaded into separate DataFrames using the provided URLs.

2. Data Information: Information about the columns, data types, and memory usage of each DataFrame is displayed.

3. Summary Statistics: Summary statistics, including count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum, are computed for each DataFrame.

4. Data Visualization: Various visualizations are created to explore the target distribution and correlations between different features. This includes bar plots for quality distribution, a heatmap for correlation analysis, scatter plots for feature-target relationships, and more.


## Model

soon :)

## References
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Modeling wine preferences by data mining from physicochemical properties." Elsevier, 47(4):547-553, 2009. ISSN: 0167-9236.
- Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

Please refer to the code files for more detailed implementation and analysis.

```ssh
git clone https://github.com/kkinastowski66/wine-quality-models.git
```
