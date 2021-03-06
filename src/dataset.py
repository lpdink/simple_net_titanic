# coding=UTF-8
import pandas as pd
import numpy as np
import tensorflow as tf
'''
第三维，Pclass:1,2,3
第五，Sex: 0/1 male=1 female=2
第六，Age: int
SibSp,Parch
Fare:float
Embarked C:1 Q:2 S:3
'''
train_csv = "../resource/data/train.csv"
test_csv = "../resource/data/test.csv"
answer_csv = "../resource/data/gender_submission.csv"

def gender_2_int(gender):
    return int(gender=="male")+1

def embarked_2_int(area):
    if area=="C":
        return 1
    elif area=="Q":
        return 2
    elif area=="S":
        return 3
    else:
        print(area)
        print("embarked_2_int warning: not_known")


class TitanicDataSet:
    def __init__(self) -> None:
        df = pd.read_csv(train_csv)
        df["Sex"]=df["Sex"].apply(lambda x:gender_2_int(x))
        df["Embarked"]=df["Embarked"].apply(lambda x: embarked_2_int(x))
        df = df[["Pclass","Survived", "Sex", "Age", "SibSp","Parch", "Fare","Embarked"]]
        # 去除一行中包含任何nan的
        df=df.dropna(axis=0,how='any')
        # 分割x和y
        x_df=df[["Pclass", "Sex", "Age", "SibSp","Parch", "Fare","Embarked"]]
        y_df=df["Survived"] 
        self.x_data = np.array(x_df)
        self.y_data = np.array(y_df)
        #print(self.x_data, self.x_data.shape)
        #print(self.y_data, self.y_data.shape)

    def get_test_set(self):
        df = pd.read_csv(test_csv)
        answer = pd.read_csv(answer_csv)

        df["Sex"]=df["Sex"].apply(lambda x:gender_2_int(x))
        df["Embarked"]=df["Embarked"].apply(lambda x: embarked_2_int(x))
        df["Survived"] = answer["Survived"]
        df = df[["Pclass","Survived", "Sex", "Age", "SibSp","Parch", "Fare","Embarked"]]
        # 用0替换所有nan
        df = df.fillna(0)
        x_df=df[["Pclass", "Sex", "Age", "SibSp","Parch", "Fare","Embarked"]]
        y_df=df["Survived"] 
        rst={
            "x_label":np.array(x_df),
            "y_label":np.array(y_df)
        }
        return rst



    def __getitem__(self, index):
        return{
            "x_label":self.x_data[index].reshape([1,-1]),
            "y_label":np.array(self.y_data[index]).reshape([1,1])
        }


if __name__=="__main__":
    data=TitanicDataSet()
    import pdb
    pdb.set_trace()
    

