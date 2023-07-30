# -*- coding: utf-8 -*-

# **Importing Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""- more libraries would be added when necessary

---

# **Importing Datasets**
"""

test = pd.read_csv("https://raw.githubusercontent.com/Ahma0205/Kaggle-competition-2208-Project/main/test.csv", index_col=[0])
train = pd.read_csv("https://raw.githubusercontent.com/Ahma0205/Kaggle-competition-2208-Project/main/train.csv", index_col=[0])
greeks =  pd.read_csv("https://raw.githubusercontent.com/Ahma0205/Kaggle-competition-2208-Project/main/greeks.csv")

"""# **EDA & Cleaning**

## general Information

### Train Dataset
"""

train.head()

train.shape

"""- we have 617 rows and 57 columns

---
"""

train.info()

"""- we have one categorical feature and some features contains null that should be handled

---

"""

null = train.isnull().sum()
null = null[null> 0]
print(null)

"""- BQ and El should be investigated more for choosing imputation method

---

### Greeks Dataset
"""

greeks.head()

greeks.shape

greeks.nunique()

greeks.describe()

greeks.info()

"""Since the Greeks metadata is only available in the training set and not in the test set, it has been decided not to include it in the models of this notebook. However, for future research, it is recommended to explore how this metadata could assist in training the model and its relationship with the features and target variable. One possible approach is to transform the binary classification problem into a multi-class classification problem by replacing the class 0 and 1 of the target variable with labels A, B, D, and G. This could provide an avenue for incorporating the Greeks data and potentially improve model performance. Further investigation in this direction may yield valuable insights.

---

## Exploring and Cleaning

### categorical feature in train dataset to numeric
"""

train['EJ'].unique()

train['EJ'] = train['EJ'].replace({'A': 0, 'B': 1})

"""- replacing hte only categorical feature to number as A to 0 and B to 1

---

"""

df= train

"""- changing the name of data set to df as "train" may be confusing after dataset split to train and test

---

### Changing the name of some feature to remove the space in the name
"""

df.columns

df.rename(columns={'BD ': 'BD', 'CD ':'CD', 'CW ': 'CW', 'FD ':'FD' }, inplace=True)

"""### distribution of feature"""

target = 'Class'

def plot_distribution(df):
    # Set the number of columns for the subplots.
    cols = 4
    # Calculate the number of rows based on the number of columns and the total number of features.
    rows = len(train.columns) // cols
    # Create a figure and a grid of subplots with the specified number of rows and columns.
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 4*rows))
    # Set the style and font scale for the plots.
    sns.set(font_scale=1, style='whitegrid')

    # Loop through each feature column in the DataFrame.
    for i, col_name in enumerate(train.columns):
        # Exclude the 'Class' column from the plot, as it is the target variable.
        if col_name != 'Class':
            # Get the current subplot.
            ax = axes[i // cols, i % cols]
            # Create a KDE plot for the current feature, separated by the target variable 'Class'.
            # The 'hue' parameter is set to 'target', which ensures that the KDE plot is colored based on the target classes.
            # The 'fill=True' option fills the area under the KDE curve.
            # The 'alpha=0.5' sets the transparency of the filled area.
            # The 'linewidth=1' adjusts the thickness of the KDE curve.
            sns.kdeplot(data=df, x=col_name, hue=target, ax=ax, fill=True, alpha=0.5, linewidth=1)

            # Set the title for the subplot, indicating the feature name and its distribution with respect to the target.
            ax.set_title(f'{col_name.title()} Distribution by {target.title()}', fontsize=10)
            # Set the x-axis label with the feature name.
            ax.set_xlabel(col_name.title(), fontsize=8)
            # Set the y-axis label with the target variable name.
            ax.set_ylabel(target.title(), fontsize=8)
            # Set the tick label size for both x-axis and y-axis.
            ax.tick_params(axis='both', which='major', labelsize=8)
            # Set the legend for the subplot, representing the target classes (0 and 1) with their titles.
            ax.legend([1, 0], title=target.title(), fontsize=8)

    # Adjust the layout and spacing between subplots for better visualization.
    plt.tight_layout()
    # Display the plot.
    plt.show()

# Call the function 'plot_distribution' with the DataFrame 'df' as input.
plot_distribution(df)

"""- Here are two important observations regarding the distributions of features:

  - The data set contains various types of distributions, and some of them may require transformation before they can be used for modeling. Since there is no specific pattern common to all of these distributions, employing Quantile Transformation, which is a general method known to work well with unspecified distributions, seems like a suitable approach.

  - It is noteworthy that the distributions of features across both target classes (0 and 1) are mostly similar. Therefore, no further effort is required to handle these distributions in the context of the target variable.

---

### corelations

#### Corelation Matrix
"""

def plot_heatmap(df, title):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    #Create a mask for the diagonal elements
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True
    # Set the colormap and figure size
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(30, 30))

    # Set the title and font properties
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=18)

    # Plot the heatmap
    sns.heatmap(corr_matrix, linewidths=0.1, vmax=1.0, vmin=-1.0,
                square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 7, "weight": "bold"}, mask=mask)

# Call the function with the DataFrame and title
plot_heatmap(train, title='Correlation Matrix')

"""- Based on the table, it is evident that certain features exhibit high correlation with each other. In light of this correlation, we have the opportunity to enhance our model by reducing the number of features that do not contribute significantly to its performance. Thus, we can consider dropping these highly correlated features, as they are not providing any additional value to our model. By doing so, we can potentially improve the model's efficiency and maintain or even enhance its predictive capability. it is decided to remove corelations more 90%

---

"""

import numpy as np

# Step 1: Calculate the correlation matrix for the DataFrame 'df'
correlation_matrix = df.corr()

# Step 2: Create a boolean mask to ignore the self-correlation
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Step 3: Apply the mask to the correlation matrix to set the self-correlation values to NaN
correlation_matrix_masked = correlation_matrix.mask(mask)

# Step 4: Find highly correlated pairs in the masked correlation matrix
threshold = 0.9
highly_correlated = [
    (column, corr_column, correlation_matrix.loc[column, corr_column])
    for column in correlation_matrix_masked.columns
    for corr_column in correlation_matrix_masked.columns
    if (column != corr_column) and
    (correlation_matrix_masked.loc[column, corr_column] > 0.9 or
     correlation_matrix_masked.loc[column, corr_column] < -0.9
    )
]

# Step 5: Print the highly correlated pairs and their correlation values
for item in highly_correlated:
    print(f"Column {item[0]} has correlation of {item[2]:.2f} with {item[1]}")

"""To avoid potential mistakes in identifying highly correlated features from the correlation matrix visually, it is advisable to programmatically specify features with correlations greater than +0.90 or less than -0.90.

---

#### Dropping highly corelated features
"""

# Drop the highly correlated columns
for item in highly_correlated:
    column_to_drop = item[0]
    if column_to_drop in df.columns:
        df.drop(column_to_drop, axis=1, inplace=True)
        print(f"Dropped column: {column_to_drop}")

"""#### Investigating corelations more

"""

corr_matrix = df.corr()
# Get pairs with correlation above 0.5 (excluding self-correlations on the diagonal)
pairs_to_plot = [(i, j) for i in range(corr_matrix.shape[0]) for j in range(i+1, corr_matrix.shape[0]) if abs(corr_matrix.iloc[i, j]) > 0.6]

# Define how many plots per line
plots_per_line = 5

# Calculate required rows for subplot
rows = np.ceil(len(pairs_to_plot) / plots_per_line)

fig, ax = plt.subplots(int(rows), plots_per_line, figsize=(plots_per_line * 5, rows * 5))

# Flat the axes for easier iteration
ax = ax.flatten()

for idx, (i, j) in enumerate(pairs_to_plot):
    ax[idx].scatter(df.iloc[:, i], df.iloc[:, j])
    ax[idx].set_xlabel(df.columns[i])
    ax[idx].set_ylabel(df.columns[j])
    ax[idx].set_title(f'Correlation: {corr_matrix.iloc[i, j]:.2f}')

# Remove unused subplots
if len(pairs_to_plot) < len(ax):
    for idx in range(len(pairs_to_plot), len(ax)):
        fig.delaxes(ax[idx])

plt.tight_layout()
plt.show()

"""- Upon further investigation into the correlation between features, it has come to light that certain features' correlations might be influenced by the presence of outliers in one or more features. Addressing these outliers could potentially lead to a more accurate representation of the underlying relationships between the features.

---

#### Investigating highly corelated features to target variable
"""

# Calculate the correlation matrix of the features
corr_matrix = df.corr()

# Identify the top 10 features most correlated with your target variable
top_corr_cols = corr_matrix["Class"].abs().nlargest(11).index

# Create a smaller correlation matrix with only these features
top_corr = corr_matrix.loc[top_corr_cols, top_corr_cols]

# Create a mask to hide the upper triangle of the correlation matrix (since it's a mirror of the lower triangle)
mask = np.zeros_like(top_corr)
mask[np.triu_indices_from(mask)] = True

# Generate the heatmap using seaborn
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(top_corr, mask=mask, annot=True, ax=ax)

# Add title and show the plot
ax.set_title('The heat map of Top 10 features correlation with the target variable')
plt.show()

"""- Considering the limited number of rows in the dataset (617 records) and the relatively large number of features (57), there is a possibility that the model might encounter the overfitting problem. To address this issue, one approach is to only include highly correlated features with the target variable in the model, rather than importing all features. However, for the time being, we will continue with the entire dataset to assess if overfitting becomes a concern or not. By monitoring the model's performance and generalization capabilities, we can gain insights into the necessity of selecting only highly correlated features to mitigate potential overfitting in future iterations.

---

### Exploring Outliers
"""

def plotbox(df, n_cols=3):

    # Set the style for the plots.
    sns.set_style('whitegrid')

    # Get the names of all columns in the DataFrame.
    cols = df.columns

    # Calculate the number of rows required based on the number of columns and the provided 'n_cols' value.
    n_rows = (len(cols) - 1) // n_cols + 1

    # Create a figure and a grid of subplots with the specified number of rows and columns.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*5))

    # If there are less than n_cols*2 plots, reshape the axes to be in a single row for better visualization.
    if len(cols) < n_cols*2:
        axes = np.reshape(axes, (-1,))

    # Loop through each numeric variable and create a boxplot in the corresponding subplot.
    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.boxplot(y=var_name, data=df, ax=ax, showmeans=True,
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"3"})
        ax.set_title(f'Boxplot of {var_name}')

    # Adjust the layout and spacing between subplots for better visualization.
    plt.tight_layout()
    # Display the plot.
    plt.show()

# Call the function with the DataFrame 'df' and specify the number of columns for subplot arrangement (n_cols=5).
plotbox(df, n_cols=5)

"""- It is evident that certain features in our dataset contain extreme outliers that need to be addressed. However, due to the limited number of rows, we must avoid removing any records to maintain the integrity of our dataset. Additionally, traditional methods such as identifying outliers using 1.5 IQR from the first and third quantiles could result in significant data alteration since we have a small dataset.

- Therefore, we have decided to take a visual approach to handle these outliers. For each feature, we will visually identify records that deviate significantly from others and replace them with the maximum value present in that specific feature. This way, we can mitigate the impact of outliers without compromising the overall size and structure of the dataset.

---

### Dealing with outlier
"""

# Identify the outliers
outliers = {
    'AB': df['AB'] > 2.5,
    'AF': df['AF'] > 15000,
    'AH': df['AH'] > 1500,
    'AM': df['AM'] > 250,
    'CH': df['CH'] > 0.15,
    'AR': df['AR'] > 75,
    'AX': df['AX'] > 15,
    'CL': df['CL'] > 8,
    'DA': df['DA'] > 125,
    'AZ': df['AZ'] > 35,
    'AY': df['AY'] > 10,
    'BC': df['BC'] > 25,
    'BD': df['BD'] > 30000,
    'CS': df['CS'] > 100,
    'BP': df['BP'] > 800,
    'BR': df['BR'] > 50000,
    'CR': df['CR'] > 2.5,
    'CB': df['CB'] > 1000,
    'CC': df['CC'] > 2,
    'CU': df['CU'] > 4,
    'DF': df['DF'] > 12,
    'DH': df['DH'] > 0.8,
    'CF': df['CF'] > 50,
    'CD': df['CD'] > 400,
    'DI': df['DI'] > 800,
    'DL': df['DL'] > 300,
    'DU': df['DU'] > 25,
    'DY': df['DY'] > 120,
    'EB': df['EB'] > 30,
    'FI': df['FI'] > 25,
    'GB': df['GB'] > 100,
    'GH': df['GH'] > 70,
    'EE': df['EE'] > 15,
    'EP': df['EP'] > 350,
    'EH': df['EH'] > 2,
    'EG': df['EG'] > 20000,
    'FL': df['FL'] > 80,
    'FS': df['FS'] > 2,
    'EU': df['EU'] > 1000,
    'FC': df['FC'] > 2500,
    'FE': df['FE'] > 100000,
    'GE': df['GE'] > 800,
    'FR': df['FR'] > 20
}

# Remove the outliers
for feature, mask in outliers.items():
    df = df.loc[~mask]

# Replace removed outliers with the maximum value for each feature
for feature in outliers.keys():
    df[feature] = df[feature].fillna(df[feature].max())

"""- our data is clean and ready to go further for more engineering and modeling steps

---

# **Feature Engineering**

## Dataset Split and Oversampling
"""

X = df.drop('Class', axis=1)
y=df['Class'].astype(int)

"""- Target variable splitted from dataset

---
"""

# Split before imputation
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

"""-
This code snippet splits the dataset into training and test sets with an 80-20 ratio, respectively. The split is performed before any scaling or transformation is applied to avoid data leakage from the training set into the test set. Data leakage can occur when information from the training set unintentionally influences the test set, leading to overly optimistic performance metrics and unrealistic model evaluation. By conducting the split prior to any data processing steps, we ensure that the two sets remain independent and representative of their respective subsets, providing a reliable basis for model training and evaluation.
Also, stratify=y is used for stratified sampling. It ensures that the target variable's distribution is preserved in both the training and testing sets. By setting stratify=y, the split will be done in a way that maintains the same proportion of class labels in both sets as seen in the original dataset.

---
"""

X_train.shape, X_test.shape, y_train.shape, y_test.shape

y.value_counts(),y_train.value_counts(),y_test.value_counts()

"""- Stratified sampling worked well

---

## Trasforming
"""

from sklearn.preprocessing import QuantileTransformer

# Create an instance of the QuantileTransformer with your desired settings.
Quantile_transformer = QuantileTransformer(n_quantiles=100, random_state=42)

#  Fit the transformer on the entire dataset to compute the quantiles and then transform the data.
transformed_data = Quantile_transformer.fit_transform(X_train)

# Step 4: Create a new DataFrame with the transformed data and column names.
X_train = pd.DataFrame(transformed_data, columns=X_train.columns)

X_test =  pd.DataFrame(Quantile_transformer.transform(X_test), columns=X_test.columns)

"""- As we mentioned before in data cleaning part, quantile tranform is suitable for our case

## Scaling
"""

from sklearn.preprocessing import StandardScaler

# Apply Normalization to the entire dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create a new DataFrame with the Normalization data
X_train= pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

# Print the transformed dataset
X_train.head()

"""- As our data has been transformed to normal form, we use standard scaler to turn them into same scale"""

X_test_scaled = scaler.transform(X_test)

# Create a new DataFrame with the Normalization data
X_test= pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

"""- After fitting the scaler with the training set, we use the same scaler to transform our test set. The scaling process is based solely on the statistics (e.g., mean and standard deviation) of the training set, and these statistics are then applied to the test set independently. This prevents any information from the test set from influencing the scaling process, maintaining the integrity of the evaluation and avoiding potential overfitting or biased results.

---

## Imputation
"""

from sklearn.impute import KNNImputer

# Create the imputer
imputer = KNNImputer(n_neighbors=5)

# Fit on the training data
imputer.fit(X_train)

# Transform both training and test data
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# convert them back to DataFrame
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

"""- KNN imputation can be effective when you have a relatively small dataset, as it relies on finding similar samples in the feature space. It can capture complex relationships and patterns in the data, making it suitable for situations where the missing data might be systematically related to other features."""

X_train.isnull().sum().sum()
X_test.isnull().sum().sum()

"""- There is not any null value in our dataset

# **Modeling**

## Oversampling
"""

from imblearn.over_sampling import RandomOverSampler

# Display the length of the original data and the proportion of each class in the target variable.
print("Length of the original data is:", len(df))
print("Proportion of True data in original data is {:.2%}".format(len(y_train[y_train == 1]) / len(y_train)))
print("Proportion of False data in original data is {:.2%}".format(len(y_train[y_train == 0]) / len(y_train)))

# Apply Random Oversampling to the training set
oversampler = RandomOverSampler(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Display the length of the oversampled data and the proportion of each class in the target variable after oversampling.
print("Length of the oversampled data is:", len(X_train))
print("Proportion of True data in oversampled data is {:.2%}".format(len(y_train[y_train == 1]) / len(y_train)))
print("Proportion of False data in oversampled data is {:.2%}".format(len(y_train[y_train == 0]) / len(y_train)))

# Plot the distribution of the target variable after oversampling.
plt.figure(figsize=(10, 6))
plt.hist(y_train, bins=[-0.5, 0.5, 1.5], rwidth=0.5, color='green', alpha=0.7)
plt.xticks([0, 1])
plt.title('Distribution of target after oversampling')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

"""- RandomOverSampler is used to balance the class distribution in the training set. After oversampling, both classes have an equal proportion of 50%. This process increases the size of the training set and improves the model's ability to learn patterns from the minority class. The resulting histogram shows equal counts for both classes, indicating a balanced dataset for training

## Model

### Importing necessary libraries for modeling
"""

#pip install catboost

from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score , roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

"""### Training initial Model"""

from sklearn.metrics import log_loss, precision_score
from sklearn.linear_model import LogisticRegression

models = {
    'Logistic regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5, metric='minkowski'),
    'Support vector kernel': SVC(kernel='rbf', random_state=42),
    'Support vector linear kernel': SVC(kernel='linear'),
    'XGBOOST': XGBClassifier(),
    'Decision tree': DecisionTreeClassifier(),
    'Random forest': RandomForestClassifier(),
    'Naive bayes': GaussianNB(),
    'catb': CatBoostClassifier(verbose=0, iterations=300, depth=6, learning_rate=0.1, loss_function='Logloss', random_seed=42),
    'MLPC': MLPClassifier(hidden_layer_sizes=(300, 600, 100), activation="relu", solver='adam', random_state=42, alpha=0.01)
}

acc_train = []  # List to store train accuracy
acc_test = []   # List to store test accuracy
log_losses = []  # List to store log loss
precisions = []  # List to store precision scores
models_names = []

for key, model in models.items():
    model.fit(X_train, y_train)
    models_names.append(key)

    # Calculate train accuracy using cross-validation
    train_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=5)) * 100
    acc_train.append(train_accuracy)

    # Calculate test accuracy
    test_accuracy = model.score(X_test, y_test) * 100
    acc_test.append(test_accuracy)

    # Calculate log loss (only for models supporting probability estimation)
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
        log_loss_value = log_loss(y_test, y_pred_prob)
        log_losses.append(log_loss_value)
    else:
        log_losses.append(None)

    # Calculate precision score
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    precisions.append(precision)

models_scores = pd.DataFrame({
    'model name': models_names,
    'train accuracy': acc_train,
    'test accuracy': acc_test,
    'log loss': log_losses,
    'precision': precisions
})

models_scores.head(10)

"""- It is evident that many of the models faced overfitting, which was expected given the small size of our dataset (only 617 records) and the limited number of samples (around 430) used for training after the train-test split. Overfitting occurs when a model becomes too tailored to the training data, capturing patterns that might not generalize well to new, unseen data. As a result, the performance on new data may be poor.

- In machine learning, the primary goal is to build models that can generalize well to new, unseen data, making accurate predictions beyond the training set. To address overfitting in our models, we can consider several approaches:

  - Use Simple Models: Instead of using complex models that might memorize the training data, opting for simpler models can reduce overfitting. Simple models have fewer parameters, making them less likely to fit noise in the data.

  - Feature Reduction or Feature Selection: Reducing the number of features or selecting the most important ones can help prevent overfitting and improve model generalization.

  - Hyperparameter Tuning: Careful tuning of hyperparameters can fine-tune the model's behavior and help find the right balance between underfitting and overfitting.

  - Ensemble Methods: Ensemble methods, such as Random Forest or Gradient Boosting, combine the predictions of multiple models to improve generalization and robustness.

- Moving forward, we will focus on tuning the models to optimize their performance. By finding the right hyperparameters , we aim to mitigate overfitting and improve the models' ability to make accurate predictions on new, unseen data.

# **Tunning**
"""

# Define the classification models
models = {
    'Random forest': RandomForestClassifier(),
    'SVC': SVC(kernel='rbf'),
    'catb': CatBoostClassifier(verbose=0, iterations=300, loss_function='Logloss', random_seed=42),
    'XGBOOST': XGBClassifier(),
    'MLPC': MLPClassifier(hidden_layer_sizes=(30, 60, 10), activation="relu", solver='adam', random_state=42)
}


param_grid = {
        'Random forest': {
        'max_depth': [3],
        'n_estimators': [ 200, 300],
        'max_leaf_nodes': [3, 5],
        "criterion": ["gini", "entropy"],
        'max_features': ['sqrt', 'log2', None]
    },

    'SVC':{
          'kernel': ['rbf', 'sigmond']
    },

    'XGBOOST': {
              'max_depth':[2],
              'gamma':[2,3],
              'eta': [0.5],
              'reg_alpha': [3],
              'reg_lambda':[3]
    },

    'catb': {
              'depth':[3],
              'learning_rate':[0.01,0.05],
    },

    'MLPC': {
              'alpha': [0.001, 0.01, 0.1]
    }

}

# Perform grid search cross-validation for each model and save the best models and parameters
best_models = {}
best_params = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    best_params[model_name] = grid_search.best_params_

# Evaluate the best models using different metrics
for idx, (model_name, model) in enumerate(best_models.items()):
    print(f'Evaluation for {model_name}:')

    # Evaluate on the train set
    y_train_pred = model.predict(X_train)
    print('Train Set:')
    print(classification_report(y_train, y_train_pred))

    # Calculate log loss on the train set (if available)
    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(X_train)
        train_log_loss = log_loss(y_train, y_train_prob)
        print(f'Train Log Loss: {train_log_loss}')

    # Evaluate on the test set
    y_test_pred = model.predict(X_test)
    print('Test Set:')
    print(classification_report(y_test, y_test_pred))

    # Calculate log loss on the test set (if available)
    if hasattr(model, 'predict_proba'):
        y_test_prob = model.predict_proba(X_test)
        test_log_loss = log_loss(y_test, y_test_prob)
        print(f'Test Log Loss: {test_log_loss}')

    # Add a dotted line after each model's result for better separation
    if idx < len(best_models) - 1:
        print('-' * 50)

# Print the best parameters for each model
for model_name, params in best_params.items():
    print(f'Best parameters for {model_name}: {params}')

"""- After tuning the models with their best parameters, we evaluated their performance on both the training and test sets. The Random Forest, CatBoost, and XGBoost models showed relatively good performance on the training set but struggled with predicting class 1 accurately on the test set, indicating possible overfitting.

- To address this issue, we decided to use an ensemble method, specifically the VotingClassifier. This method combines the predictions of multiple base classifiers (Random Forest, CatBoost, and XGBoost) using hard voting to make the final prediction. By leveraging the strengths of each model, the ensemble can potentially mitigate overfitting and improve the overall performance on class 1 predictions.

- Further experimentation with different ensemble combinations and hyperparameters can help optimize the ensemble's performance and ensure better generalization on the specific dataset.

#**Evaluation models with Best parameters**
"""

from sklearn.metrics import confusion_matrix

# Assuming the best parameters for Random Forest and XGBoost models are stored in these variables.
best_Rf_params = best_params['Random forest']
best_SVC_params = best_params['SVC']
best_XGB_params = best_params['XGBOOST']
best_MLPC_params = best_params['MLPC']
best_catb_params = best_params['catb']

models = {
    'Random forest': RandomForestClassifier(**best_Rf_params),
    'SVC': SVC(**best_SVC_params),
    'XGBOOST': XGBClassifier(**best_XGB_params),
    'MLPC': MLPClassifier(hidden_layer_sizes=(300, 600, 100), activation="relu", solver='adam', random_state=42, alpha=0.01),
    'catb': CatBoostClassifier(verbose=0, iterations=300, loss_function='Logloss', random_seed=42),


}


# Create a function to plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, ax):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(type(model).__name__)

# Create a figure with multiple subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 3))

# Fit the models on the training data
for model in models.values():
    model.fit(X_train, y_train)

# Plot confusion matrix for each model
for i, model in enumerate(models.values()):
    plot_confusion_matrix(model, X_test, y_test, axes[i])

# Adjust the layout of subplots
plt.tight_layout()

# Show the plot
plt.show()

"""- The confusion matrix visualization provides an overview of the model's performance, showing how well it predicts true positive, false positive, true negative, and false negative values. This helps in evaluating and comparing the models' predictive abilities.
- comparing overall, performance of each model on confusion matrix it seems that MLPC has the best performance. but as each model have its own strenght, ensemble method mat aggregate the result to a better prediction

## Features Importance
"""

# Create a function to plot feature importance
def plot_feature_importance(model, ax):
    # Get feature importance
    importances = model.feature_importances_

    # Get feature names from your data
    feature_names = X_train.columns

    # Create a DataFrame for features and their importance
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # Sort DataFrame by importance
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    # Plot feature importance
    sns.barplot(data=importance_df.head(10), x='importance', y='feature', ax=ax, orient='h')
    ax.set_title(f'{type(model).__name__}')

# Create a figure with multiple subplots
fig, axes = plt.subplots(1, len(models), figsize=(10, 3), sharey='row')

# Fit the models on the training data and plot feature importance
for i, (model_name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    if model_name in ['Random forest', 'XGBOOST','catb' ]:
        plot_feature_importance(model, axes[i])

# Adjust the layout of subplots
plt.tight_layout()

# Show the plot
plt.show()

"""# **Ensemble Methods**

## Stacking Classifier
"""

from sklearn.ensemble import StackingClassifier

# Assuming the best parameters for each model are stored in these variables.
best_Rf_params = best_params['Random forest']
best_SVC_params = best_params['SVC']
best_CatBoost_params = best_params['catb']
best_XGBOOST_params = best_params['XGBOOST']
best_MLPC_params = best_params['MLPC']

# Define the classification models with their best parameters
models = {
    'Random forest': RandomForestClassifier(**best_Rf_params),
    'SVC': SVC(**best_SVC_params),
    'catb': CatBoostClassifier(**best_CatBoost_params, verbose=0, iterations=300, loss_function='Logloss', random_seed=42),
    'XGBOOST': XGBClassifier(**best_XGBOOST_params),
    'MLPC': MLPClassifier(**best_MLPC_params, hidden_layer_sizes=(300, 600, 100), activation='relu', solver='adam', random_state=42)

    }
# Initialize the StackingClassifier with a LogisticRegression meta-model
stacking_clf = StackingClassifier(estimators=list(models.items()), final_estimator=LogisticRegression())

# Fit the StackingClassifier on the training data
stacking_clf.fit(X_train, y_train)

# Evaluate on the train set
y_train_pred = stacking_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_classification_report = classification_report(y_train, y_train_pred)

# Calculate log loss on the train set (if available)
if hasattr(stacking_clf, 'predict_proba'):
    y_train_prob = stacking_clf.predict_proba(X_train)
    train_log_loss = log_loss(y_train, y_train_prob)
else:
    train_log_loss = None

# Evaluate on the test set
y_test_pred = stacking_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_report = classification_report(y_test, y_test_pred)

# Calculate log loss on the test set (if available)
if hasattr(stacking_clf, 'predict_proba'):
    y_test_prob = stacking_clf.predict_proba(X_test)
    test_log_loss = log_loss(y_test, y_test_prob)
else:
    test_log_loss = None

# Print the evaluation metrics
print("Evaluation Metrics:")
print("Train Accuracy:", train_accuracy)
print("Train Classification Report:")
print(train_classification_report)

if train_log_loss is not None:
    print("Train Log Loss:", train_log_loss)

print("\nTest Accuracy:", test_accuracy)
print("Test Classification Report:")
print(test_classification_report)

if test_log_loss is not None:
    print("Test Log Loss:", test_log_loss)

"""- With this stacking we could improve the performance of the model to 88% fro class 1 in test set. we are going to test different ensemble methods to see the results

## Voting Classifier
"""

from sklearn.ensemble import VotingClassifier

# Assuming you have defined X_train, y_train, X_test, y_test, and best_params for each model as before.

# Define the classification models with their best parameters
models = {
    'Random forest': RandomForestClassifier(**best_Rf_params),
    'SVC': SVC(**best_SVC_params),
    'catb': CatBoostClassifier(**best_CatBoost_params, verbose=0, iterations=300, loss_function='Logloss', random_seed=42),
    'XGBOOST': XGBClassifier(**best_XGBOOST_params),
    'MLPC': MLPClassifier(**best_MLPC_params, hidden_layer_sizes=(300, 600, 100), activation='relu', solver='adam', random_state=42)
}

# Initialize the VotingClassifier with the models
voting_clf = VotingClassifier(estimators=list(models.items()), voting='hard')  # 'hard' voting means majority rule

# Fit the VotingClassifier on the training data
voting_clf.fit(X_train, y_train)

# Evaluate on the train set
y_train_pred = voting_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_classification_report = classification_report(y_train, y_train_pred)

# Calculate log loss on the train set (if available)
if hasattr(voting_clf, 'predict_proba'):
    y_train_prob = voting_clf.predict_proba(X_train)
    train_log_loss = log_loss(y_train, y_train_prob)
else:
    train_log_loss = None

# Evaluate on the test set
y_test_pred = voting_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_report = classification_report(y_test, y_test_pred)

# Calculate log loss on the test set (if available)
if hasattr(voting_clf, 'predict_proba'):
    y_test_prob = voting_clf.predict_proba(X_test)
    test_log_loss = log_loss(y_test, y_test_prob)
else:
    test_log_loss = None

# Print the evaluation metrics
print("Evaluation Metrics:")
print("Train Accuracy:", train_accuracy)
print("Train Classification Report:")
print(train_classification_report)

if train_log_loss is not None:
    print("Train Log Loss:", train_log_loss)

print("\nTest Accuracy:", test_accuracy)
print("Test Classification Report:")
print(test_classification_report)

if test_log_loss is not None:
    print("Test Log Loss:", test_log_loss)

"""- voting classifier could not improve the result. So Stacking classifiers would be considered as the final model of this notebook.

- The feature importance analysis reveals that both CatBoost and Random Forest heavily rely on the "CR" feature to make predictions. However, XGBoost shows a more balanced consideration of various features, suggesting that it might be a better model in terms of utilizing the entire feature set for its predictions.

# **Conclusion**

- Even though significant improvements have been made in model performance through various methods such as feature engineering, tuning, and ensembles, overfitting is still an issue due to the small dataset size. Additionally, the model's difficulty in predicting class 1 is a notable concern.

- To address this, the recommendation is to conduct further research and exploration specifically focusing on class 1 of the dataset. Some possible approaches are suggested:

  - Utilize Metadata: Consider incorporating metadata into the analysis. Adding relevant metadata features might help improve predictions for class 1.

  - Multiple Models for Class 1: Instead of predicting class 1 as a single entity, consider breaking it down into multiple subgroups and using separate models for each subgroup. This might lead to better predictions for class 1.

  - Investigate Metadata-Class Relationship: Investigate the relationship between metadata and the target class. Analyzing how metadata is related to class 1 could offer insights into improving predictions.

-

```
# This is formatted as code
```

By implementing these suggestions, future research can aim to further enhance the model's performance, particularly in predicting class 1, and address the challenges posed by the small dataset size.

# **Submission to Kaggle competition**

## cleaning and feature engineering
"""

testcopy=test.copy()
test

test.isnull().sum().sum()

test['EJ'] = test['EJ'].replace({'A': 0, 'B': 1})

test.rename(columns={'BD ': 'BD', 'CD ':'CD', 'CW ': 'CW', 'FD ':'FD' }, inplace=True)

test.drop(['BZ','DV', 'FD','GL'] ,axis=1,inplace=True)

test_scaled =scaler.transform(test)

test= pd.DataFrame(test_scaled, columns=test.columns, index=test.index)

"""- the same step and transformers in training used to get the test set ready for the prediction

---

## prediction with final model
"""

selected_model = stacking_clf

# The 'predict' method takes the test data as input and returns the predicted target values for each sample.
testcopy['prediction'] = selected_model.predict(test)

# The 'predict_proba' method takes the test data as input and returns the predicted probabilities for each class.
testcopy[['predict_0_proba','predict_1_proba']]=selected_model.predict_proba(test)

testcopy
