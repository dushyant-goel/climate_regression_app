from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

st.title("🌤️ Climate Regression at Heathrow")
st.subheader("""
Explore changes in temperature, sunshine hours, and frost days over time using regression models.
""")

st.markdown(f"""
In this report we will investigate weather data collected at the Met office station at Heathrow. The data has 8 features, namely 
- `yyyy` : year, range is 1957 to 2024
- `mm` : month, Number 1-12
- `t_max` : Mean daily maximum temperature for the month
- `t_min` : Mean daily minimum temperature for the month
- `af_days` : Days of air frost for the month
- `sun_hours` : Hours of sunshine for the month

We want to predict the `sun_hours` and `af_days` from the `t_max` and `t_min`.
For this we will use supervised learning method **linear regression**.  
              
We have about 920 rows or feature vectors in our total data set. 
This is a compartively small dataset. In case of climate data, there is an 
expectation of linearity. Warmer months will raise both the maximum and minimum
mean temprature. Number of hours of sunshine should be able to be modeled 
linearly with both these features.
  
Linear Regression is a simple model and easy to interprete. It is 
computationaly less intensive. It can easily be extended to polynomial 
regression. At worst, it provides as baseline to compare to against other more 
complex techniques."""
            )

# Load data function


@st.cache_data
def load_weather_data():
    file_path = './data/heathrowdata.txt'
    column_names = ['yyyy', 'mm', 't_max', 't_min',
                    'af_days', 'rain_mm', 'sun_hours', 'extra']

    # Read file and clean it
    df = pd.read_csv(file_path, skiprows=7, sep=r'\s+', names=column_names)
    df.drop(columns=['extra'], inplace=True)
    df.replace('---', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Clean sun_hours column and convert types
    df['sun_hours'] = df['sun_hours'].str.replace('#', '', regex=False)

    numeric_columns = ['yyyy', 'mm', 't_max',
                       't_min', 'af_days', 'rain_mm', 'sun_hours']
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors='coerce')

    return df


# Load and display data
weather_data = load_weather_data()

st.markdown("## 📄 Raw Heathrow Weather Data (Cleaned)")

st.dataframe(weather_data.head(10))

st.markdown(
    f"**Total Rows:** {weather_data.shape[0]} | **Columns:** {weather_data.shape[1]}")

"---"

st.markdown("## Theory")
st.markdown(r"""
Any supervised machine learning method, like regression, requires a set of labled data 
""")

st.latex(r"""
D := \{(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n))\}
""")

st.markdown(r"""
Here, $X_i^d$ are the features and $Y_i$ is the label, d is dimension.
For a regression model, the label $Y_i \in \mathbb{R}$ is a continuous variable.  
Suppose $X$ is a $d$-dimensional vector, then a model $\phi(X)$ maps:           
""")
st.latex(r"""            
\mathbb{R}^d \rightarrow \mathbb{R}
""")

st.markdown("#### Joint Probability $P$, unknown")
st.markdown(r"""
The variables $X$ and $Y$ are dependent (if $X_i$ are to have any predictive influence on $Y$).  
We assume that $(X, Y)$ follow some joint distribution $P$, which is unknown.

Our aim is to find $\phi(X)$ such that it mimics this joint distribution:
""")

st.latex(r"""
P(X) = Y \Rightarrow \phi(X) \approx Y
""")

st.html(r"""
        <blockquote>
        <p><b>
How do we calculate φ(X) ?
        </b></p>
        </blockquote>
            """)

st.markdown(r"""
We first divide the data $D$ into two parts:  
- $D_{train}$:  training data  
- $D_{test}$:   testing data  

Initially, we use a split 70:30.
""")

st.markdown("#### $\phi(X)$ 'learns' $P$")
st.markdown(r"""
We want to "learn" a model $\phi(X) \rightarrow Y$.

We assume a supervised learning method $\phi$ with some initial parameters and "fit" or adjust these using training data.  
We then test the model on unseen data to evaluate its performance. How is this done?
""")

st.markdown(r"""
Since we only observe $P(X) = Y$ in the training set, we fit $\phi$ to match those observations using
a key statistic: **Root Mean Squared Error (RMS)**. It is defined as the mean of sum of difference between true
and predicted labels:
""")


st.latex(r"""
RMS(\phi) = \frac{1}{n} \sum_{i=1}^{n} (\phi(X_i) - Y_i)^2
""")
st.markdown("***$\phi$ uses RMS is to tune the parameters***")

st.markdown(r"""
Where:
- $Y_i$: true label (`true_label`)  
- $\phi(X_i)$: predicted label (`predicted_label`)  

RMS is referred to as the **training error** or empirical error.
""")

"---"

st.markdown("## Regression Model $\phi(X)$")
st.markdown("#### Equation of Line")
st.markdown("""
A linear regression model outputs $\phi$, which represents a linear surface through the $d$-dimensional space of feature vector $X$.  
The (hyper-)line is such that the (euclidean-)distance of all the points $x_i$ from it is minimized.
""")
st.markdown("Recall that in 2D, a line is represented as:")

st.latex(r"""
y = mx + m_0
""")

st.markdown("""
Where:  
- $m$ is the slope  
- $m_0$ is the intercept
""")


st.markdown("#### Line in $D$-Dimensions")

st.markdown(r"""
We generalize this equation to $D$-dimensions.  
Let """)
st.latex(r"""
            x = [x^1, x^2, \ldots, x^d], 
""")
st.markdown(r"""
a feature vector (data-point) of dimension $d$. 
Then the slope vector is """)
st.latex(r"""
            m = [m^1, m^2, \ldots, m^d],      
""")
st.markdown(r"""
and similarly the intercept vector is:""")
st.latex(r"""
          m_0 = [m^1_0, m^2_0, \ldots, m^d_0]
""")
st.markdown(r"""
In ML $m$ is called the weight and is denoted by $w$.
$\phi(x)$ is given by the product $m \cdot x$, which becomes:
""")

st.latex(r"""
\phi(X) = Y_{\text{pred}} = w^T X + w_0
""")


st.markdown("#### Finding $w$")

st.markdown("""
The model $\phi(x)$ has unknown parameters $w^j$, for $0 \leq j \leq d$.  
We estimate these by minimizing RMS error.

Given a dataset $D_{\text{train}}$ with $n$ samples, $0 \leq i \leq n$:

""")

st.latex(r"""
\hat{R}(\phi) = \frac{1}{n} \sum_{i=1}^n \left( \phi(X_i) - Y_i \right)^2
""")

st.markdown("Substituting $\phi(X)$:")

st.latex(r"""
\hat{R}(\phi) = \frac{1}{n} \sum_{i=1}^n \left( w^T X_i + w_0 - Y_i \right)^2
""")


st.markdown("#### Ordinary Least Squares (OLS)")

st.markdown("""
To minimize the above quadratic error expression, we differentiate:
""")

st.latex(r"""
\frac{\partial \hat{R}}{\partial w} = 0, \quad \frac{\partial \hat{R}}{\partial w_0} = 0
""")

st.markdown("This leads to the optimal solution:")
st.latex(r"""
w = \Sigma_{Y,X} \Sigma_{X,X}^{-1}
""")
st.latex(r"""
w_0 = \bar{Y} - w \bar{X}^T
""")


st.markdown("Where the covariance matrices are defined as:")

st.latex(r"""
\Sigma_{X,X} = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^T (X_i - \bar{X})
""")
st.latex(r"""
\Sigma_{Y,X} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \bar{Y})^T (X_i - \bar{X})
""")
st.markdown("""
Here, $\Sigma_{X,X}$ and $\Sigma_{Y,X}$ represent covariance terms:  
- $\Sigma_{X,X}$: covariance of features with themselves  
- $\Sigma_{Y,X}$: covariance between target $Y$ and features $X$
""")

"---"

# Explore and Visualize data
st.markdown("## Data Exploration")

st.markdown(f"""
Now that the data is cleaned, we begin by exploring it visually and statistically.  
We will group data by year or month and explore relationships between temperature (`t_max`, `t_min`) and targets (`sun_hours`, `af_days`).  
""")


st.markdown("#### 📈 Annual Mean Temperatures with trend line")
st.markdown("We compute yearly average for `t_max` and `t_min` to observe trend over the years."
            "We observe a general warming trend in the maximum and minimum temperatures at Heathrow.")

grouped_by_year = weather_data.groupby('yyyy').mean(numeric_only=True)

years = grouped_by_year.index.values
t_max = grouped_by_year['t_max'].values
t_min = grouped_by_year['t_min'].values

fig, ax = plt.subplots(figsize=(10, 5))

sns.set_style("whitegrid")
sns.regplot(x=years, y=t_max, label='Max Temp (°C)', color='red',
            scatter_kws={'s': 40}, line_kws={'color': 'darkred', 'linestyle': '--'}, ax=ax)
sns.regplot(x=years, y=t_min, label='Min Temp (°C)', color='blue',
            scatter_kws={'s': 40}, line_kws={'color': 'darkblue', 'linestyle': '--'}, ax=ax)

ax.set_title("Yearly Mean Temperatures at Heathrow (1957-2024)")
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)


st.markdown("#### 🌤️ Monthly Means: Sunshine & Frost")
st.markdown(
    "We compute monthly averages across years for `sun_hours` and `af_days` to understand seasonality.")

grouped_by_month = weather_data.groupby('mm').mean(numeric_only=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
grouped_by_month[['sun_hours', 'af_days']].plot(ax=ax2)
ax2.set_title("Average Monthly Sunshine Hours & Air Frost Days")
ax2.set_xlabel("Month")
ax2.set_ylabel("Hours / Days")
st.pyplot(fig2)

st.markdown("#### 📈 Boxplot: Max Temperature Distribution by Month")

st.markdown("""
This plot shows the distribution of maximum daily temperatures (`t_max`) across all years, grouped by calendar month.  
It reveals seasonality — warmer months (July, August) have higher medians and wider spreads.
""")

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='mm', y='t_max', data=weather_data, ax=ax, palette=None)
ax.set_title("Monthly Distribution of Max Temperature (°C)")
ax.set_xlabel("Month")
ax.set_ylabel("Max Temperature (°C)")
st.pyplot(fig)


# Regression Analysis


"---"

st.markdown("## 🔁 Multi-Feature Linear Regression")

st.markdown(r"""
The feature vector $X$ is a 4-dimensional vector with the following components:

1. $X^1$: Max Temperature (`t_max`)  
2. $X^2$: Min Temperature (`t_min`)  
3. $X^3$: Year (`yyyy`)  
4. $X^4$: Month (`mm`)

We perform two experiments to predict continuous labels:

1. $Y$: Sunshine Hours (`sun_hours`)  
2. $Y$: Air Frost Days (`af_days`)

Year and Month are included to capture seasonality and long-term warming trends observed in earlier sections.
""")

# Let user choose which Y to predict
target_variable = st.selectbox("Choose target variable (Y):", [
                               'sun_hours', 'af_days'])

# Feature matrix and label
X = weather_data[['t_max', 't_min', 'yyyy', 'mm']]
Y = weather_data[target_variable]

# Split 70:30
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# RMS Error
rms = np.sqrt(mean_squared_error(Y_test, Y_pred))
st.markdown(f"**RMS Error on Test Set:** `{rms:.2f}`")

# Predicted vs True Scatter Plot
fig, ax = plt.subplots()
ax.scatter(Y_test, Y_pred, alpha=0.6, color='blue',
           s=30, label='Predicted vs True')
ax.plot([Y_test.min(), Y_test.max()], [
        Y_test.min(), Y_test.max()], 'r--')  # Identity line
ax.set_xlabel("True Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"Prediction vs Actual for {target_variable}")
st.pyplot(fig)

st.markdown(r"""
    The regression line for air frost days is effected by the skewed spread of 
    data points. A lot of months have no air frost days, clustering points at "0".
""")

# Model Evaluation Section

"---"

st.markdown("## 📊 Model Evaluation Across Training Sizes")

st.markdown(r"""
We evaluate model performance using **Root Mean Squared Error (RMS)** on test data, but varying the size of training
data.
""")

st.html("""
        <blockquote>
        <p>
How would decreasing the training set size improve the prediction? 
Because the model may overfit to the training data, leading to lower RMS on training but higher RMS on test data.
Additionally, adding more data points have diminishing returns on model performance, as we will see in the plots below.
        </p>
        </blockquote>
""")

st.markdown("""
Let:
- $\phi(X)$: our trained model
- $Y$: true label
- ${RMS}_{test} = E[(\phi(X) - Y)^2]$

A smaller ${RMS}_{test}$ indicates that the model $\phi(X)$ better approximates the true joint distribution $P(X, Y)$.

We vary the **training set size from 10% to 70%**, using a fixed test set of 30%, and track performance for:
1. `sun_hours` as target
2. `af_days` as target

""")


def train_test_model(X, y, test_size, train_size, random_state=42):
    # Split the data into train and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Select only part of the training data
    split_idx = int(train_size * len(X_train_full))
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rms_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rms_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return rms_train, rms_test


def evaluate_model(X, y, test_size=0.3):
    train_sizes = np.linspace(0.1, 0.7, 10)  # Training sizes from 10% to 70%
    rms_train, rms_test = [], []

    for train_size in train_sizes:
        rms_train_i, rms_test_i = train_test_model(X, y, test_size, train_size)
        rms_train.append(rms_train_i)
        rms_test.append(rms_test_i)

    return train_sizes, rms_train, rms_test


st.markdown("#### 🧪 Learning Curves")

# Prepare features and targets
X = weather_data[['t_max', 't_min', 'yyyy', 'mm']]
y_sun = weather_data['sun_hours']
y_af = weather_data['af_days']

# Evaluate both targets
train_sizes_sun, rms_train_sun, rms_test_sun = evaluate_model(
    X, y_sun, test_size=0.3)
train_sizes_af, rms_train_af, rms_test_af = evaluate_model(
    X, y_af, test_size=0.3)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# sun_hours plot
axes[0].plot(train_sizes_sun, rms_train_sun,
             label='Train RMS', color='blue', marker='o')
axes[0].plot(train_sizes_sun, rms_test_sun,
             label='Test RMS', color='red', marker='o')
axes[0].set_title('Learning Curve: sun_hours')
axes[0].set_xlabel('Training Set Fraction')
axes[0].set_ylabel('RMS')
axes[0].legend()

# af_days plot
axes[1].plot(train_sizes_af, rms_train_af,
             label='Train RMS', color='blue', marker='o')
axes[1].plot(train_sizes_af, rms_test_af,
             label='Test RMS', color='red', marker='o')
axes[1].set_title('Learning Curve: af_days')
axes[1].set_xlabel('Training Set Fraction')
axes[1].set_ylabel('RMS')
axes[1].legend()

st.pyplot(fig)

st.markdown("#### 📖 Interpretation of Learning Curves")

st.markdown(r"""
The **learning curves** above show how the model performance (measured by RMS) varies as we change the **training set size** from 10% to 70%.

- The **test RMS** first decreases, then flattens — indicating the point upto which adding more training data helps. Since we find an elbow in both
the target labels, the models are at generalization capacity. 

**For `sun_hours`**:
- The **training RMS** increases steadily as training data increases and then flatlines. Initially, the model overfits on training data, but performs poorly
on test data. As the data size grows, the model becomes more general as indicated by decreasing test RMS. 

**For `af_days`**
- The **training RMS** decreases steadily as training data size increase, but the test RMS flatlines. Because a large number of targets are 0,
air frost days' predictions are difficult to generalize. Hence, we don't see an RMS train descreasing. 

""")

# Correlation

"---"

st.markdown("## 🔍 Feature Analysis and Dimensionality Considerations")

st.markdown(r"""
In the previous section, we evaluated the model's ability to learn using **as much data as possible**, and observed the RMS performance as training size increased.

Having reached the limit of what more data can offer, we now turn to **feature engineering** — to understand whether **all inputs are necessary**, or if some can be **safely removed**.

We begin with a **correlation and covariance analysis** between the input features and the target variables (`sun_hours`, `af_days`). This helps us identify:
- Features that are strongly correlated (and may be redundant)
- Inputs that dominate the prediction signal (e.g. `t_max` for `sun_hours`)
- Opportunities to reduce the **dimensionality** of our model

> ⚠️ **Note**: This analysis is useful for **interpretability**, **computation**, and **regularization**, but **does not drastically reduce RMS test error** on its own.
""")

st.markdown("""
We compute Pearson correlation coefficients between the numerical variables in the dataset.  
""")

# Compute correlation matrix
corr_matrix = weather_data[['t_max', 't_min', 'sun_hours', 'af_days']].corr()

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", square=True, ax=ax)
ax.set_title("Correlation Matrix: Key Variables")
st.pyplot(fig)

st.markdown("""
    We observe that `t_max` and `t_min` are highly correlated. We can safely collapse them into a single variable. 
    `t_max` is more strongly correlated with `sun_hours` and `t_min` with `af_days`. We will use this fact in the following sections.
""")

"---"

# Ridge Regression

st.markdown("## 📏 Ridge Regression with λ (Lambda) Tuning")

# Theory Recap

st.markdown(r"""
Linear models like OLS are sensitive to outliers and multicollinearity in input features. 
Linear regression performs poorly when:  
- Features are correlated (e.g., `t_max` and `t_min`).
- Model overfits on outliers or noise.
To mitigate this, **Ridge Regression** introduces a penalty term proportional to the magnitude of model weights:
""")

st.latex(r"""
\hat{R}(\phi) = \frac{1}{n} \sum_{i=1}^n \left( \phi(X_i) - Y_i \right)^2 + \lambda \|w\|^2
""")

st.markdown(r"""
This **regularizes** the model, stabilizes training when features are correlated, and controls overfitting.  
We evaluate the model for different values of $\lambda$, and report the **test RMS**.
""")


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_ridge_regression(X, y, lambdas, test_size=0.3, random_state=42):
    results = {'lambda': [], 'rms_train': [], 'rms_test': []}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    for l in lambdas:
        model = Ridge(alpha=l)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results['lambda'].append(l)
        results['rms_train'].append(
            root_mean_squared_error(y_train, y_train_pred))
        results['rms_test'].append(
            root_mean_squared_error(y_test, y_test_pred))

    return pd.DataFrame(results)


st.markdown("#### 🔬 Experiment Setup")

st.markdown("""
We remove:
- `t_min` when predicting `sun_hours`
- `t_max` when predicting `af_days`.  
based on previous covariance observations.
""")

lambdas = [0.1, 1, 10, 100, 1000, 10000]

# sun_hours prediction
X_sun = weather_data[['t_max', 'mm', 'yyyy']]
y_sun = weather_data['sun_hours']
sun_results = evaluate_ridge_regression(X_sun, y_sun, lambdas)

# af_days prediction
X_af = weather_data[['t_min', 'mm', 'yyyy']]
y_af = weather_data['af_days']
af_results = evaluate_ridge_regression(X_af, y_af, lambdas)

st.markdown("#### 📊 Results: RMS vs λ")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sun Hours
axes[0].plot(sun_results['lambda'], sun_results['rms_test'],
             marker='o', color='teal')
axes[0].set_xscale('log')
axes[0].set_xlabel('Lambda (log scale)')
axes[0].set_ylabel('Test RMS')
axes[0].set_title('Ridge Regression — Sun Hours')

# Air Frost Days
axes[1].plot(af_results['lambda'], af_results['rms_test'],
             marker='o', color='indigo')
axes[1].set_xscale('log')
axes[1].set_xlabel('Lambda (log scale)')
axes[1].set_ylabel('Test RMS')
axes[1].set_title('Ridge Regression — Air Frost Days')

st.pyplot(fig)

st.markdown("#### 🧠 Interpretation")

st.markdown(r"""
  - RMS initially decreases as $\lambda$ increases → regularization helps prevent overfitting.
  - After an "elbow", RMS increases again → important signal is suppressed.
  - Suggests an optimal $\lambda$ in range (e.g., 100-1000).
  - **No clear loss from removing removing input feature `t_max` and `t_min` from training of `af_days` and `sun_hours`
    respectively — test RMS remains similar despite removal**. 

""")

"---"

# Validation Section

st.markdown("## 🔁 K-Fold Cross-Validation for Ridge Regression")

st.markdown(r"""
Since our dataset is relatively small, the performance metrics can vary depending on how the data is split.  
To reduce this variability, we use **K-Fold Cross-Validation** — training and validating the model on multiple data splits.

We perform this for Ridge Regression with varying values of $\lambda$, chosen around the 'elbow' point identified earlier.

Since there was no clear reduction in RMS_test for `af_days`, we only perform this validation on `sun_hours`.
""")


def cross_validate_ridge(X, y, l, k=5, random_state=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    train_rms_list = []
    test_rms_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Ridge(alpha=l)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rms_list.append(root_mean_squared_error(y_train, y_train_pred))
        test_rms_list.append(root_mean_squared_error(y_test, y_test_pred))

    return {
        'train_rms_mean': np.mean(train_rms_list),
        'train_rms_std': np.std(train_rms_list),
        'test_rms_mean': np.mean(test_rms_list),
        'test_rms_std': np.std(test_rms_list),
    }


st.markdown("#### 🔬 Practical Setup")

st.markdown(r"""
We focus on predicting **Sun Hours**, using:

- Features: `t_max`, `mm`, `yyyy`  
- Targets: `sun_hours`  
- K-Folds: 5  
- Lambdas: 20 values between $10^{1.9}$ and $10^{3.1}$ (log-spaced)

This will help validate if our lambda tuning generalizes across multiple data slices.
""")

# Define search space
lambdas = np.logspace(1.9, 3.1, 20)
X_cv = weather_data[['t_max', 'mm', 'yyyy']]
y_cv = weather_data['sun_hours']

results = []
for chosen_lambda in lambdas:
    res = cross_validate_ridge(X_cv, y_cv, l=chosen_lambda, k=5)
    results.append(res)

# Extract metrics
train_means = [r['train_rms_mean'] for r in results]
train_stds = [r['train_rms_std'] for r in results]
test_means = [r['test_rms_mean'] for r in results]
test_stds = [r['test_rms_std'] for r in results]

st.markdown("#### 📈 Cross-Validation Results")

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(lambdas, test_means, yerr=test_stds,
            fmt='-o', capsize=5, label='Test RMS')
ax.set_xscale('log')
ax.set_xlabel("Lambda (log scale)")
ax.set_ylabel("RMS Error")
ax.set_title("Ridge Regression Cross-Validation (Sun Hours)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("#### 🧠 Interpretation")

st.markdown(r"""
- As seen earlier, Ridge Regression **slightly improves** model performance around $\lambda \approx 100$ to  $\lambda \approx 500$
- However, the improvements are **not dramatic**, and the elbow is shallow
- Cross-validation confirms that the improvements seen earlier weren't just due to a lucky train/test split
- This suggests the model is **relatively stable**.

This validates the robustness of our feature choices and the modeling pipeline.
""")

"---"

st.markdown("## ✅ Summary and Reflections")

st.markdown(r"""
In this project, we investigated **climate patterns at Heathrow** using **linear and ridge regression** to predict:
- ☀️ Monthly **sunshine hours**
- ❄️ Monthly **air frost days**

#### 🔍 Key Takeaways:
- **Clear warming trend** observed in both minimum and maximum temperatures over time.
- Linear regression performs reasonably well for both targets, especially `sun_hours`.
- **Learning curves** show that increasing training data helps up to a point — after which performance flattens.
- Feature analysis reveals:
  - `t_max` and `t_min` are highly correlated
  - `t_max` is more relevant for `sun_hours`, `t_min` for `af_days`
- **Ridge regression** slightly improves generalization and helps reduce overfitting, especially for `sun_hours`.
- **Cross-validation** confirms the robustness of the model and stability of selected hyperparameters.

#### 📘 Reflection:
This project highlights the value of:
- Careful **data cleaning and feature selection**
- Using **learning curves** and **cross-validation** to evaluate model performance
- Combining **statistical insight** with **ML workflows** for interpretable, stable models


📤 The full Streamlit app is now available on GitHub.
🛠️ [Link](https://github.com/dushyant-goel/climate_regression_app)

""")
