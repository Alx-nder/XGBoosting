
import pandas as pd
import urllib.request
import zipfile

url="https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip"
fname='kaggle-survey-2018.zip'
member_name='multipleChoiceResponses.csv'

def extract_zip(src,dest, member_name):
    url=src
    fname=dest
    fin=urllib.request.urlopen(url)
    data=fin.read()

    with open(dest,mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dest) as z:
        kag=pd.read_csv(z.open(member_name))
        kag_question=kag.iloc[0]
        raw=kag.iloc[1:]
        return raw

raw = extract_zip(url,fname,member_name)

def tweak_kag(df_:pd.DataFrame)-> pd.DataFrame:
    return(df_
           .assign(age=df_['Q2'].str.slice(0,2).astype(int),
                   education=df_['Q4'].replace({"Master’s degree":18,"Bachelor’s degree":16,"Doctoral degree":20,"Some college/university study without earning a bachelor’s degree":13,"Professional degree":19,"I prefer not to answer":None,"No formal education past high school":12}),
                   major=(df_['Q5']
                          .pipe(topn,n=3)
                          .replace({'Computer science (software engineering, etc.)':'cs',
                                    "Engineering (non-computer focused)":'eng',
                                    "Mathematics or statistcs":'stat',})
                                    ),
                    years_exp=(df_['Q8'].str.replace('+','',regex=False)
                               .str.split('-',expand=True)
                               .iloc[:,0]
                               .astype(float)
                               ),
                    compensation=(df_['Q9']
                                .str.replace('+','',regex=False)
                                .str.replace(',','',regex=False)
                                .str.replace('500000','500',regex=False)
                                .str.replace('I do not wish to disclose my approximate yearly compensation','0',regex=False)
                                .str.split('-',expand=True)
                                .iloc[:,0]
                                .fillna(0)
                                .astype(int)
                                .mul(1_000)
                                ),
                    python=df_['Q16_Part_1'].fillna(0).replace('Python',1),
                    r=df_['Q16_Part_2'].fillna(0).replace('R',1),
                    sql=df_['Q16_Part_3'].fillna(0).replace('SQL',1)
                    )#end assign
                .rename(columns=lambda col:col.replace(' ','_'))
                .loc[:,'Q1,Q3,age,education,major,years_exp,compensation,python,r,sql'.split(',')]
    

    )
def topn(ser,n=5,default='other'):
    counts=ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]),default)

from feature_engine import encoding,imputation
from sklearn import base,pipeline

class TweakKagTransformer(base.BaseEstimator,base.TransformerMixin):
    def __inint__(self,ycol=None):
        self.ycol=ycol
    def transform(self,X):
        return tweak_kag(X)
    def fit(self,X,y=None):
        return self



def get_rawX_y(df,col_y):
    raw=(df.query('Q3.isin(["United States of America","China","India"]) and Q6.isin(["Data Scientist","Software Engineer"])'))
    return raw.drop(columns=[col_y]) , raw[col_y]

# pipeline
kag_pl=pipeline.Pipeline([('tweak',TweakKagTransformer()),('cat',encoding.OneHotEncoder(top_categories=5,drop_last=True,variables=['Q1','Q3','major'])),('num_impute',imputation.MeanMedianImputer(imputation_method='median',variables=['education','years_exp']))]
)

from sklearn import model_selection
kag_X,kag_y=get_rawX_y(raw,'Q6')

kag_X_train,kag_X_test,kag_y_train,kag_y_test=model_selection.train_test_split(kag_X,kag_y,test_size=.3,random_state=42,stratify=kag_y)

Xtrain=kag_pl.fit_transform(kag_X_train,kag_y_train)
Xtest=kag_pl.transform(kag_X_test)
print(Xtrain)


