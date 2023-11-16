import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os, time

def preview_dataframe(df,data):
    print("PEREVIEW DATA")
    print(df,"\n")

    print("DATA SET INFOMATION")
    print(df.info(),"\n")

    print("DATA SHAPE")
    print(df.shape,"\n")

    df_description = pd.read_csv(os.path.join(root_path,data["Disease Description"]["csv"]))
    print("DATA DESCRIPTION")
    print(df_description,"\n")


def data_processing(dataframe,data,seed):
    dataframe = shuffle(dataframe,random_state=seed)

    for col in dataframe.columns:
        dataframe[col] = dataframe[col].str.replace('_',' ')

    #Tạo một biến cols và gán giá trị là danh sách các tên cột của DataFrame df.
    cols = dataframe.columns 

    # Tạo một biến data và gán giá trị là một mảng 1D chứa toàn bộ dữ liệu của DataFrame df. df[cols] được sử dụng để chọn các cột được định nghĩa bởi cols, và .values.flatten() chuyển đổi thành mảng 1D.
    temp_data = dataframe[cols].values.flatten()

    # Tạo một Series s từ mảng 1D data. Mỗi giá trị trong mảng sẽ trở thành một phần tử trong Series.
    s = pd.Series(temp_data)
    s = s.str.strip() 

    # chuyển đổi lại Series s thành một mảng đa chiều với kích thước giống với kích thước của DataFrame df ban đầu bằng cách sử dụng phương thức reshape. Mục đích là tái tạo lại cấu trúc của DataFrame ban đầu sau khi đã thực hiện các bước trước đó.
    s = s.values.reshape(dataframe.shape)
    dataframe = pd.DataFrame(s, columns=dataframe.columns)
    dataframe = dataframe.fillna(0)

    df_symptom_severity = pd.read_csv(os.path.join(root_path, data["Symptom Severity"]["csv"]))
    df_symptom_severity['Symptom'] = df_symptom_severity['Symptom'].str.replace('_',' ')
    df_symptom_severity['Symptom'].unique()
    vals = dataframe.values
    symptoms = df_symptom_severity['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df_symptom_severity[df_symptom_severity['Symptom'] == symptoms[i]]['weight'].values[0]
        
    d = pd.DataFrame(vals, columns=cols)
    d = d.replace('dischromic  patches', 0)
    d = d.replace('spotting  urination',0)
    dataframe = d.replace('foul smell of urine',0)
    def note():
            # vals = df.values: Tạo một mảng NumPy vals chứa giá trị của DataFrame df. Mảng này sẽ được sử dụng để thực hiện các thay thế.
            # symptoms = df1['Symptom'].unique(): Tạo một mảng chứa các giá trị duy nhất từ cột 'Symptom' trong DataFrame df1. Điều này giúp xác định các triệu chứng duy nhất trong DataFrame df1.
            # Vòng lặp for:
            # for i in range(len(symptoms)):: Lặp qua từng phần tử trong mảng symptoms.
            # vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]: Thay thế các giá trị trong vals bằng giá trị của cột 'weight' từ DataFrame df1 tương ứng với triệu chứng symptoms[i].
            # d = pd.DataFrame(vals, columns=cols): Tạo DataFrame mới d từ mảng vals, với các cột được giữ nguyên từ DataFrame df.
        pass

    return dataframe

def null_checker(df):
    # Tạo một DataFrame trống để lưu trữ kết quả
    result = pd.DataFrame(columns=['count'])
    
    # Duyệt qua từng cột của DataFrame và tính số lượng giá trị null
    for col in df.columns:
        count_null = sum(df[col].isnull())
        result.loc[col] = [count_null]
    return result


def Decision_Tree_MODEL(df, x_train, y_train, x_test, y_test, criterion:str, max_depth, seed, save_model_path:str):
    repo = {}
    start_time = time.time()

    # Mô hình descition tree
    tree = DecisionTreeClassifier(criterion=criterion,random_state=seed,max_depth = max_depth)
    tree.fit(x_train, y_train)
    end_time = time.time()
    preds = tree.predict(x_test)
    training_time = end_time - start_time

    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())

    kfold = KFold(n_splits=10,shuffle = True,random_state=seed)
    DS_train = cross_val_score(tree, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_train,columns=['Scores'])
    DS_test = cross_val_score(tree, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_test,columns=['Scores'])

    Standard_deviation = DS_test.std()
    Accuracy_score = DS_test.mean()
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="RdPu", linewidths=.5, cbar_kws={"shrink": 0.8})
    heatmap.set_title("Confusion Matrix Heatmap use Decision Tree Classifier")
    plt.show()

    repo["Predictions"] = preds
    repo["Confusion Matrix"] = conf_mat
    repo["Model"] = tree
    repo["Training time"] = training_time
    repo["Heatmap Confusion Matrix"] = sns.heatmap(df_cm)
    repo["Data Score train"] = DS_train
    repo["Data Score test"] = DS_test
    repo["Accuracy"] = Accuracy_score
    repo["Standard Deviation"] = Standard_deviation

    if save_model_path:
        joblib.dump(tree, save_model_path)
        print(f'Model saved at: {save_model_path}')

    return repo

def Random_Forest_MODEL(df, x_train, y_train, x_test, y_test, max_features, max_depth, seed, save_model_path:str):
    repo = {}
    start_time = time.time()
    rnd_forest = RandomForestClassifier(random_state=seed, max_features = max_features, n_estimators= 500, max_depth = max_depth)
    rnd_forest.fit(x_train,y_train)
    end_time = time.time()
    preds=rnd_forest.predict(x_test)
    training_time = end_time - start_time

    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())

    kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
    rnd_forest_train = cross_val_score(rnd_forest, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(rnd_forest_train,columns=['Scores'])
    rnd_forest_test = cross_val_score(rnd_forest, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(rnd_forest_test,columns=['Scores'])

    sns.heatmap(df_cm)
    Accuracy_score = rnd_forest_train.mean()
    Standard_deviation = rnd_forest_train.std()

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    heatmap.set_title("Confusion Matrix Heatmap use Random Forest Classifier")
    plt.show()

    repo["Predictions"] = preds
    repo["Confusion Matrix"] = conf_mat
    repo["Model"] = rnd_forest
    repo["Training time"] = training_time
    repo["Heatmap Confusion Matrix"] = sns.heatmap(df_cm)
    repo["Data Score train"] = rnd_forest_train
    repo["Data Score test"] = rnd_forest_test
    repo["Accuracy"] = Accuracy_score
    repo["Standard Deviation"] = Standard_deviation

    if save_model_path:
        joblib.dump(rnd_forest, save_model_path)
        print(f'Model saved at: {save_model_path}')

    return repo


if __name__ == "__main__":
    SEED = 2005
    root_path = os.path.dirname(os.path.abspath(__file__))

    files = ["dataset.csv",
                "dataset.json",
                "disease_Description.csv",
                "disease_Description.json",
                "Symptom-severity.csv",
                "Symptom-severity.json",
                "disease_precaution.csv",
                "disease_precaution.json"]
    file_path = [os.path.join("dataset",file) for file in files]
    data = {
            "Data set" :{
                "csv":  file_path[0],
                "json": file_path[1],
            },           
            "Disease Description":{
                "csv":  file_path[2],
                "json": file_path[3],
            },
            "Symptom Severity":{
                "csv":  file_path[4],
                "json": file_path[5],
            },
            "Disease Precaution":{
                "csv":  file_path[6],
                "json": file_path[7],
            },
    }   

    df = pd.read_csv(os.path.join(root_path, data["Data set"]["csv"]))
    df_symptom_severity = pd.read_csv(os.path.join(root_path, data["Symptom Severity"]["csv"]))

    preview_dataframe(df,data)

    df = data_processing(df,data,SEED)

    # preview_dataframe(df,data)

    print("Number of symptoms used to identify the disease ",len(df_symptom_severity['Symptom'].unique()))
    print("Number of diseases that can be identified ",len(df['Disease'].unique()))

    features = df.iloc[:,1:].values
    labels = df['Disease'].values

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size = 0.4,random_state=SEED)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    print("\nUSE - DESCISION TREE MODEL")
    Model_report_DS = Decision_Tree_MODEL(df,x_train,y_train,x_test,y_test,criterion='entropy',max_depth=10,seed=SEED, save_model_path= "DescisionTreeModel.joblib")
    Model_report_DS["Heatmap Confusion Matrix"]

    print(f"Training time: {Model_report_DS['Training time']*1000:.3f} ms")
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (Model_report_DS["Accuracy"]*100.0, Model_report_DS["Standard Deviation"]*100.0))

    print("\nUSE - RANDOM FOREST CLASSIFER MODEL")
    Model_report_RFC = Random_Forest_MODEL(df, x_train, y_train, x_test, y_test, max_features = 'sqrt', max_depth = 10, seed = SEED,save_model_path= "RandomForestModel.joblib")
    print(f"Training time: {Model_report_RFC['Training time']*1000:.3f} ms")
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (Model_report_RFC["Accuracy"]*100.0, Model_report_RFC["Standard Deviation"]*100.0))
    

    disease_description = pd.read_csv(os.path.join(root_path, data["Disease Description"]["csv"]))
    disease_precaution = pd.read_csv(os.path.join(root_path, data["Disease Precaution"]["csv"]))

    algorithms = ('Decision Tree', 'Random Forest')

    train_accuracy = (Model_report_DS["Data Score train"].mean()*100.0,
                    Model_report_RFC["Data Score train"].mean()*100.0,)
    
    test_accuracy = (Model_report_DS["Data Score test"].mean()*100.0,
                    Model_report_RFC["Data Score test"].mean()*100.0,)
    
    test_standard_deviation = (Model_report_DS["Data Score test"].std()*100.0,
                    Model_report_RFC["Data Score test"].std()*100.0,)
    
    # create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    index = np.arange(len(algorithms))
    bar_width = 0.3
    opacity = 1
    rects1 = plt.bar(index, train_accuracy, bar_width, alpha = opacity, color='Cornflowerblue', label='Train')
    rects2 = plt.bar(index + bar_width, test_accuracy, bar_width, alpha = opacity, color='Teal', label='Test')
    rects3 = plt.bar(index + bar_width, test_standard_deviation, bar_width, alpha = opacity, color='red', label='Standard Deviation')

    plt.xlabel('Algorithm') # x axis label
    plt.ylabel('Accuracy (%)') # y axis label
    plt.ylim(0, 115)
    plt.title('Comparison of Algorithm Accuracies') # plot title
    plt.xticks(index + bar_width * 0.5, algorithms) # x axis data labels
    plt.legend(loc = 'upper right') # show legend

    for index, data in enumerate(train_accuracy):
        plt.text(x = index - 0.035, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    for index, data in enumerate(test_accuracy):
        plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    for index, data in enumerate(test_standard_deviation):
        plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    plt.show()

    