import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('attr_training.csv')

df['Holiday'] = df['Holiday'].fillna(0)
df.loc[df.Holiday != 0, 'Holiday'] = 1
df['Weeks'] = pd.to_datetime(df['Weeks'])
df = pd.get_dummies(df, columns=['Category'])
df['Sales_Lift'] = df['Incremental_Sales'] / df['Dollar_Sales']
df = df[df['Tactic_ID'] > 0]

X1 = df['Tactic_Total_Cost'].values
Y1 = df['Sales_Lift'].values
X = X1.reshape(-1, 1)
Y = Y1.reshape(-1, 1)
reg = LinearRegression().fit(X, Y)
print(reg.coef_)
print(reg.intercept_)

df = df[['Tactic_ID', 'Weeks', 'Holiday', 'Dollar_Sales', 'Base_Dollar_Sales',
         'Incremental_Sales', 'Sales_Lift', 'Act_Impressions',
         'Category_Digital Banner Ad', 'Category_Digital Bloggers/Influencers',
         'Category_Digital Email', 'Category_Digital Social Media',
         'Category_In-Store POS', 'Category_Retailer-Led Tactics',
         'Insertion_Cost', 'Redemption_Cost', 'Tactic_Total_Cost']]

print(df.describe(include='all'))

# df_2 = df.groupby(['Tactic_ID', 'Weeks'])['Sales_Lift'].first().reset_index()
#
# plt.scatter(df_2['Weeks'].values, df_2['Sales_Lift'].values)
# plt.show()

with pm.Model() as model:
    std = pm.Uniform("std", 0, 100)

    beta = pm.Normal("beta", mu=12, sd=10)
    alpha = pm.Normal("alpha", mu=101, sd=10)

    mean = pm.Deterministic("mean", alpha + beta * X)

    obs = pm.Normal("obs", mu=mean, sd=std, observed=Y)

    trace = pm.sample(10000, step=pm.Metropolis())
    burned_trace = trace[20000:]

pm.plots.traceplot(burned_trace, varnames=["std", "beta", "alpha"])

pm.plot_posterior(burned_trace, varnames=["std", "beta", "alpha"])

std_trace = burned_trace['std']
beta_trace = burned_trace['beta']
alpha_trace = burned_trace['alpha']

pd.Series(std_trace[:1000]).plot()

std_mean = std_trace.mean()
beta_mean = beta_trace.mean()
alpha_mean = alpha_trace.mean()
