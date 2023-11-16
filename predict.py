import joblib,os,time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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

    for col in dataframe.columns:
        dataframe[col] = dataframe[col].str.replace('_',' ')

    cols = dataframe.columns
    temp_data = dataframe[cols].values.flatten()

    s = pd.Series(temp_data)
    s = s.str.strip()
    s = s.values.reshape(dataframe.shape)
    dataframe = pd.DataFrame(s, columns=dataframe.columns)
    def note1():
        #  cols = df.columns: Tạo một biến cols và gán giá trị là danh sách các tên cột của DataFrame df.
        # data = df[cols].values.flatten(): Tạo một biến data và gán giá trị là một mảng 1D chứa toàn bộ dữ liệu của DataFrame df. df[cols] được sử dụng để chọn các cột được định nghĩa bởi cols, và .values.flatten() chuyển đổi thành mảng 1D.
        # s = pd.Series(data): Tạo một Series s từ mảng 1D data. Mỗi giá trị trong mảng sẽ trở thành một phần tử trong Series.
        # s = s.str.strip(): Dùng phương thức strip() để loại bỏ khoảng trắng ở đầu và cuối của mỗi chuỗi trong Series s.
        # s = s.values.reshape(df.shape): Chuyển đổi lại Series s thành một mảng đa chiều với kích thước giống với kích thước của DataFrame df ban đầu bằng cách sử dụng phương thức reshape. Mục đích là tái tạo lại cấu trúc của DataFrame ban đầu sau khi đã thực hiện các bước trước đó.
        pass
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
    def note2():
            # vals = df.values: Tạo một mảng NumPy vals chứa giá trị của DataFrame df. Mảng này sẽ được sử dụng để thực hiện các thay thế.
            # symptoms = df_symptom_severity['Symptom'].unique(): Tạo một mảng chứa các giá trị duy nhất từ cột 'Symptom' trong DataFrame df_symptom_severity. Điều này giúp xác định các triệu chứng duy nhất trong DataFrame df_symptom_severity.
            # Vòng lặp for:
            # for i in range(len(symptoms)):: Lặp qua từng phần tử trong mảng symptoms.
            # vals[vals == symptoms[i]] = df_symptom_severity[df_symptom_severity['Symptom'] == symptoms[i]]['weight'].values[0]: Thay thế các giá trị trong vals bằng giá trị của cột 'weight' từ DataFrame df_symptom_severity tương ứng với triệu chứng symptoms[i].
            # d = pd.DataFrame(vals, columns=cols): Tạo DataFrame mới d từ mảng vals, với các cột được giữ nguyên từ DataFrame df.
        pass

    return dataframe

def predict_by_DT(Model,x_test,y_test,seed):
    report = {}
    loaded_model_DT = joblib.load(Model)
    start_time = time.time()
    predict_label = loaded_model_DT.predict(x_test)
    end_time = time.time()
    predict_time = end_time - start_time

    conf_mat = confusion_matrix(y_test, predict_label)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())

    kfold = KFold(n_splits=10,shuffle = True,random_state=seed)
    DS_test = cross_val_score(loaded_model_DT, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_test,columns=['Scores'])

    Standard_deviation = DS_test.std()
    Accuracy_score = DS_test.mean()

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="RdPu", linewidths=.5, cbar_kws={"shrink": 0.8})
    heatmap.set_title("Confusion Matrix Heatmap use Decision Tree Classifier")
    plt.show()

    report["Predictions"] = predict_label
    report["Confusion Matrix"] = conf_mat
    report["Model"] = loaded_model_DT
    report["Training time"] = predict_time
    report["Heatmap Confusion Matrix"] = sns.heatmap(df_cm)
    report["Data Score test"] = DS_test
    report["Accuracy"] = Accuracy_score
    report["Standard Deviation"] = Standard_deviation

    return report

def predict_by_RF(Model,x_test,y_test,seed):
    report = {}
    loaded_model_DT = joblib.load(Model)
    start_time = time.time()
    predict_label = loaded_model_DT.predict(x_test)
    end_time = time.time()
    predict_time = end_time - start_time

    conf_mat = confusion_matrix(y_test, predict_label)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())

    kfold = KFold(n_splits=10,shuffle = True,random_state=seed)
    DS_test = cross_val_score(loaded_model_DT, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_test,columns=['Scores'])

    Standard_deviation = DS_test.std()
    Accuracy_score = DS_test.mean()

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    heatmap.set_title("Confusion Matrix Heatmap use Random Forest Classifier")
    plt.show()

    report["Predictions"] = predict_label
    report["Confusion Matrix"] = conf_mat
    report["Model"] = loaded_model_DT
    report["Training time"] = predict_time
    report["Heatmap Confusion Matrix"] = sns.heatmap(df_cm)
    report["Data Score test"] = DS_test
    report["Accuracy"] = Accuracy_score
    report["Standard Deviation"] = Standard_deviation

    return report


def diagnose_disease(Model, df_symptom_severity, disease_description, disease_precaution,symptoms:list ):
    for i in range(17-len(symptoms)):
        symptoms.append(0)
    psymptoms = symptoms
    a = np.array(df_symptom_severity["Symptom"])
    b = np.array(df_symptom_severity["weight"])

    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    psy = [np.array(psymptoms)]
    pred2 = Model.predict(psy)
    disp = disease_description[disease_description['Disease']==pred2[0]]
    disp = disp.values[0][1]
    recomnd = disease_precaution[disease_precaution['Disease']==pred2[0]]
    c = np.where(disease_precaution['Disease']==pred2[0])[0][0]
    precuation_list=[]

    for i in range(1,len(disease_precaution.iloc[c])):
          precuation_list.append(disease_precaution.iloc[c,i])
    print("The Disease Name: ",pred2[0])
    print("The Disease Discription: ",disp)
    print("Recommended Things to do at home: ")
    for i in precuation_list:
        print(i)

if __name__ == "__main__":

    SEED = 1806
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
    disease_description = pd.read_csv(os.path.join(root_path, data["Disease Description"]["csv"]))
    disease_precaution = pd.read_csv(os.path.join(root_path, data["Disease Precaution"]["csv"]))

    # preview_dataframe(df,data)

    df = data_processing(df,data,SEED)

    features = df.iloc[:,1:].values
    labels = df['Disease'].values

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.1, random_state=SEED)

    print(type(x_test))

    model_DT = "DescisionTreeModel.joblib"
    model_RF = "RandomForestModel.joblib"
    Model_descision_tree = predict_by_DT(model_DT,x_test,y_test,SEED)
    Model_random_forest = predict_by_DT(model_RF,x_test,y_test,SEED)

    sympList = df_symptom_severity["Symptom"].to_list()

    print("We can diagnose your disease based on your symptoms!")
    print("How many symptoms are you suffering from?")
    number_of_symptom = int(input("The number of symptoms you are suffering from: "))
    list_symptoms = list(disease_description.get("Disease"))
    for i in range(len(list_symptoms)):
        print(f"{i+1}. {list_symptoms[i]}")
    print("Choose your symptoms by the number of symptoms.")
    List_of_patient_symptoms = []
    for i in range(number_of_symptom):
        List_of_patient_symptoms.append(int(input(f"Symptom {i+1}: ")))

    print(len(List_of_patient_symptoms))
    diagnose_disease(Model_random_forest["Model"],df_symptom_severity, disease_description, disease_precaution,List_of_patient_symptoms)