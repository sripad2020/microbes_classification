import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
data=pd.read_csv('microbes.csv')
print(data)
print(data.isna().sum())
print(data.info())
print(data.columns)
num=data.select_dtypes(include='number').columns.values
print(num)
#for i in num:
#    sn.boxplot(data[i])
#    plt.show()
col=data.select_dtypes(include='number').columns.values
#for i in col:
#    for j in col:
#        plt.plot(data[i].head(10000),label=f'{i}',marker='o')
#        plt.plot(data[j].head(10000),label=f'{j}',marker='o')
#        plt.title(f'the {i} vs {j}')
#        plt.legend()
#        plt.show()

sn.countplot(data['microorganisms'])
plt.show()

data['z-scores']=(data.Eccentricity-data.Eccentricity.mean())/(data.Eccentricity.std())
df=data[(data['z-scores'] >-3)&(data['z-scores']<3)]
colms=data.columns.values
q1=df.Eccentricity.quantile(0.25)
q3=df.Eccentricity.quantile(0.75)
iqr=q3-q1
up=q3+1.5*iqr
lo=q1-1.5*iqr
df=df[(df.Eccentricity <up)&(df.Eccentricity > lo)]

df['z-scores']=(df.EquivDiameter-df.EquivDiameter.mean())/(df.EquivDiameter.std())
df=df[(df['z-scores'] >-3)&(df['z-scores']<3)]
print(data.EquivDiameter.shape)
q_1=df.EquivDiameter.quantile(0.25)
q_3=df.EquivDiameter.quantile(0.75)
i_qr=q_3-q_1
upp=q_3+1.5*i_qr
low=q_1-1.5*i_qr
df=df[(df.EquivDiameter <upp)&(df.EquivDiameter > low)]

df['z-scores']=(df.FilledArea-df.FilledArea.mean())/(df.FilledArea.std())
df=df[(df['z-scores'] >-3)&(df['z-scores']<3)]

qu_1=df.FilledArea.quantile(0.25)
qu_3=df.FilledArea.quantile(0.75)
i_q_r=qu_3-qu_1
upper=qu_3+1.5*i_q_r
lower=qu_1-1.5*i_q_r
df=df[(df.FilledArea <upper)&(df.FilledArea > lower)]

df['z-scores']=(df.Extent-df.Extent.mean())/(df.Extent.std())
df=df[(df['z-scores'] >-3)&(df['z-scores']<3)]

qua1=df.Extent.quantile(0.25)
qua3=df.Extent.quantile(0.75)
iq_r=qua3-qua1
upper_=qua3+1.5*iq_r
lower_=qua1-1.5*iq_r
df=df[(df.Extent <upper_)&(df.Extent > lower_)]
print(df.Extent.shape)

df['z-score']=(df.Extent-df.Extent.mean())/(df.Extent.std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quan1=df.Extent.quantile(0.25)
quan3=df.Extent.quantile(0.75)
Iqr=quan3-quan1
upper_bo=quan3+1.5*Iqr
lower_bo=quan1-1.5*Iqr
df=df[(df.Extent <upper_bo)&(df.Extent > lower_bo)]



df['z-score']=(df['EulerNumber']-df['EulerNumber'].mean())/(df['EulerNumber'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quant1=df['EulerNumber'].quantile(0.25)
quant3=df['EulerNumber'].quantile(0.75)
IQr=quant3-quant1
upper_bou=quant3+1.5*IQr
lower_bou=quant1-1.5*IQr
df=df[(df['EulerNumber'] <upper_bou)&(df['EulerNumber'] > lower_bou)]

df['z-score']=(df['BoundingBox3']-df['BoundingBox3'].mean())/(df['BoundingBox3'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quanti1=df['BoundingBox3'].quantile(0.25)
quanti3=df['BoundingBox3'].quantile(0.75)
IQ_r=quanti3-quanti1
upper_boun=quanti3+1.5*IQ_r
lower_boun=quanti1-1.5*IQ_r
df=df[(df['BoundingBox3'] <upper_boun)&(df['BoundingBox3'] > lower_boun)]

df['z-score']=(df['BoundingBox4']-df['BoundingBox4'].mean())/(df['BoundingBox4'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quantile1=df['BoundingBox4'].quantile(0.25)
quantile3=df['BoundingBox4'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['BoundingBox4'] <upper_bound)&(df['BoundingBox4'] > lower_bound)]

df['z-score']=(df['MajorAxisLength']-df['MajorAxisLength'].mean())/(df['MajorAxisLength'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quantile1=df['MajorAxisLength'].quantile(0.25)
quantile3=df['MajorAxisLength'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['MajorAxisLength'] <upper_bound)&(df['MajorAxisLength'] > lower_bound)]

df['z-score']=(df['MinorAxisLength']-df['MinorAxisLength'].mean())/(df['MinorAxisLength'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quantile1=df['MinorAxisLength'].quantile(0.25)
quantile3=df['MinorAxisLength'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['MinorAxisLength'] <upper_bound)&(df['MinorAxisLength'] > lower_bound)]

df['z-score']=(df['Perimeter']-df['Perimeter'].mean())/(df['Perimeter'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]
quantile1=df['Perimeter'].quantile(0.25)
quantile3=df['Perimeter'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['Perimeter'] <upper_bound)&(df['Perimeter'] > lower_bound)]

df['z-score']=(df['Perimeter']-df['Perimeter'].mean())/(df['Perimeter'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]
quantile1=df['Perimeter'].quantile(0.25)
quantile3=df['Perimeter'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['Perimeter'] <upper_bound)&(df['Perimeter'] > lower_bound)]

df['z-score']=(df['Perimeter']-df['Perimeter'].mean())/(df['Perimeter'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]
quantile1=df['Perimeter'].quantile(0.25)
quantile3=df['Perimeter'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['Perimeter'] <upper_bound)&(df['Perimeter'] > lower_bound)]

df['z-score']=(df['ConvexArea']-df['ConvexArea'].mean())/(df['ConvexArea'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]
quantile1=df['ConvexArea'].quantile(0.25)
quantile3=df['ConvexArea'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['ConvexArea'] <upper_bound)&(df['ConvexArea'] > lower_bound)]

df['z-score']=(df['ConvexArea']-df['ConvexArea'].mean())/(df['ConvexArea'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]
quantile1=df['ConvexArea'].quantile(0.25)
quantile3=df['ConvexArea'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['ConvexArea'] <upper_bound)&(df['ConvexArea'] > lower_bound)]

df['z-score']=(df['Area']-df['Area'].mean())/(df['Area'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]

quantile1=df['Area'].quantile(0.25)
quantile3=df['Area'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['Area'] <upper_bound)&(df['Area'] > lower_bound)]

df['z-score']=(df['raddi']-df['raddi'].mean())/(df['raddi'].std())
df=df[(df['z-score'] >-3)&(df['z-score']<3)]
quantile1=df['raddi'].quantile(0.25)
quantile3=df['raddi'].quantile(0.75)
IQR=quantile3-quantile1
upper_bound=quantile3+1.5*IQR
lower_bound=quantile1-1.5*IQR
df=df[(df['raddi'] <upper_bound)&(df['raddi'] > lower_bound)]

num=df.select_dtypes(include='number').columns.values
#for i in num:
#    sn.boxplot(df[i])
#    plt.show()
#filled area,bb3,bb4,peri,conv
x=df[['Unnamed: 0', 'Solidity', 'Eccentricity', 'EquivDiameter', 'Extrema',
       'FilledArea', 'Extent', 'Orientation', 'EulerNumber', 'BoundingBox1',
       'BoundingBox2','ConvexHull1','ConvexHull2', 'ConvexHull3', 'ConvexHull4','Centroid1', 'Centroid2','Area','raddi']].head(21778)
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['mic']=lab.fit_transform(data['microorganisms'])
y=data['mic'].head(21778)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.tree import DecisionTreeClassifier
tre=DecisionTreeClassifier()
tre.fit(x_train,y_train)
print(tre.score(x_test,y_test))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))