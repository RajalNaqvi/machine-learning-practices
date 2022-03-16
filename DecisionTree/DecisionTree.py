import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import pydotplus

class MakeDecision:
    def __init__(self,csv_filename: str, 
                  x: list,
                  y: str,
                  output_image: str,
                  ):
        self.x = x
        self.y = None
        self.data = None
        
        X = self.load_data(csv_filename,x,y)    
        X = self.label_encode(X)
        self.predict(X,output_image)
        
    def load_data(self,csv_filename: str, 
                  x: list,
                  y: str):      
        data = pd.read_csv(csv_filename, delimiter=",")
        self.data = data
        X = data[x].values
        self.y = data[y]
        return X

    def label_encode(self,X):
        from sklearn import preprocessing
        
        le_sex = preprocessing.LabelEncoder()
        le_sex.fit(['F','M'])
        X[:,1] = le_sex.transform(X[:,1]) 


        le_BP = preprocessing.LabelEncoder()
        le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
        X[:,2] = le_BP.transform(X[:,2])


        le_Chol = preprocessing.LabelEncoder()
        le_Chol.fit([ 'NORMAL', 'HIGH'])
        X[:,3] = le_Chol.transform(X[:,3]) 
        
        return X
    
    def predict(self,X,output_image):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            self.y,
                                                            test_size=0.3,
                                                            random_state=3)

        Dtree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
        trained_classifier = Dtree.fit(X_train,y_train)
        prediction = trained_classifier.predict(X_test)
        print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, prediction))
        
        from  io import StringIO
        io_data = StringIO()        
        out=tree.export_graphviz(trained_classifier,
                                 feature_names=self.x,
                                 out_file=io_data,
                                 class_names= np.unique(y_train),
                                 filled=True,
                                 special_characters=True,
                                 rotate=False) 
         
        graph = pydotplus.graph_from_dot_data(io_data.getvalue())  
        graph.write_png(output_image)
        img = mpimg.imread(output_image)
        plt.figure(figsize=(100, 200))
        plt.imshow(img,interpolation='nearest')

if __name__ == "__main__":
    MakeDecision('data_drugs.csv',
                 ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
                 'Drug',
                 'new.png')