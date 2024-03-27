# Wine-Quality
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# вспомогательные функции для построения графиков

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , fill=True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_distribution_2(df, var, target):
    grid = sns.jointplot(x=df[var], y=df[target], kind='reg')
    grid.fig.set_figwidth(8)
    grid.fig.set_figheight(8)
    plt.show()

def plot_boxplot(df, var, target):
    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df[target], y=df[var], whis=1.5)
    plt.show()

def plot_correlation_map( df ):
    corr = df.corr()
    corr = np.round(corr, 2)
    corr[np.abs(corr) < 0.3] = 0
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr,
        cmap = cmap,
        square=True,
        cbar_kws={ 'shrink' : .9 },
        ax=ax,
        annot = True,
        annot_kws = { 'fontsize' : 12 }
    )

df = pd.read_csv('/content/winequalityN.csv')

df.head(5)

df.describe()

df.isnull().sum()

## важные элементы подсвечиваются красным

plot_correlation_map(df)

columns_to_plot = list(df.columns)[1:-1]
num_columns = len(columns_to_plot)

for i, col in enumerate(columns_to_plot):
    plot_distribution_2(df, var=col, target='quality')

plt.tight_layout()
plt.show()

for i, col in enumerate(columns_to_plot):
    plot_distribution(df, var=col, target='quality', row = 'type')

plt.tight_layout()
plt.show()

sns.pairplot(df, plot_kws={'alpha': 0.6});

list(df.columns)[1:-1]

for i, col in enumerate(columns_to_plot):
    plot_boxplot(df, var=col, target='quality')

plt.tight_layout()
plt.show()

df_new = df.dropna()
df_new = df_new.drop('total sulfur dioxide', axis = 1)

le = LabelEncoder()
df_new["type"] = le.fit_transform(df_new["type"])

x = df_new.iloc[:,:-1].values
y = df_new.iloc[:,[-1]].values

ros = RandomOverSampler()
x_data,y_data = ros.fit_resample(x,y)

ss = StandardScaler()
x_scaled = ss.fit_transform(x_data)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_data,test_size=0.2,random_state=42)
