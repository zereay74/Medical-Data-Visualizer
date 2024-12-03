import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2 Add an overweight column to the data. To determine if a person is overweight, 
# first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. 
# If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the 
# value 1 for overweight
df['overweight'] = df['weight']/(df['height']/100)**2
df['overweight'] = (df['overweight'] > 25).astype(int)

# 3 Normalize data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1,
#  set the value to 0. If the value is more than 1, set the value to 1.
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)


def draw_cat_plot():
    # 5 Create a DataFrame for the cat plot 
    # using pd.melt with values from cholesterol, gluc, smoke, alco, active, and overweight
    df_cat = pd.melt(
        df,
        id_vars=['cardio'], 
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6 Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature.
    # You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # 7 Convert the data into long format and create a chart that shows the value counts of the 
    # categorical features using the following method provided by the seaborn library import: sns.catplot().
    fig = sns.catplot(
        data=df_cat, 
        x='variable', 
        y='total', 
        hue='value', 
        col='cardio', 
        kind='bar', 
        height=5, 
        aspect=1
    ).fig

    # 8 Get the figure for the output and store it in the fig variable.
    # (Already handled above when assigning the `fig` attribute of the catplot result)

    # 9 Do not modify the next two lines.
    fig.savefig('catplot.png')
    
    return fig

# 10 Draw the Heat Map in the draw_heat_map function.
def draw_heat_map():
    # 11 Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 Plot the heatmap
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap='coolwarm', 
        annot=True, 
        fmt='.1f', 
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}, 
        ax=ax
    )

    # 16 Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig