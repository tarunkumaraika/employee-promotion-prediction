from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
import pickle

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})



def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'train.csv'
    df = pd.read_csv(path, nrows=101)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def Machinelearning(request):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import sklearn
    import imblearn
    from sklearn.ensemble import RandomForestClassifier
    plt.rcParams['figure.figsize'] = (16, 5)
    plt.style.use('fivethirtyeight')

    train = pd.read_csv(r'media\train.csv')
    test = pd.read_csv(r'media\test.csv')

    print("Shape of the Training Data :", train.shape)
    print("Shape of the Test Data :", test.shape)

    train.columns
    test.columns
    train.head()
    test.head()
    train.tail()
    test.tail()
    train['department'].value_counts()
    train['region'].value_counts()
    train.describe().style.background_gradient(cmap = 'copper')
    train.describe(include = 'object')
    # lets check the Target Class Balance
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.style.use('fivethirtyeight')
    plt.subplot(1, 2, 1)
    sns.countplot(train['is_promoted'],)
    plt.xlabel('Promoted or Not?', fontsize = 10)
    plt.subplot(1, 2, 2)
    train['is_promoted'].value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%.2f%%', startangle = 90,
                                        labels = ['1','0'], shadow = True, pctdistance = 0.5)
    plt.axis('off')
    plt.suptitle('Target Class Balance', fontsize = 15)
    plt.show()

    import pandas as pd
    common_columns = train.columns.intersection(test.columns)
    train_total = train[common_columns].isnull().sum()
    test_total = test[common_columns].isnull().sum()
    train_percent = ((train_total / train.shape[0]) * 100).round(2)
    test_percent = ((test_total / test.shape[0]) * 100).round(2)
    missing_data = pd.concat([train_total, train_percent, test_total, test_percent],
                            axis=1, 
                            keys=['Train_Total', 'Train_Percent %', 'Test_Total', 'Test_Percent %'])
    missing_data.style.bar(color='gold')

    # lets impute the missing values in the Training Data
    train['education'] = train['education'].fillna(train['education'].mode()[0])
    train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].mode()[0])
    # lets check whether the Null values are still present or not?
    print("Number of Missing Values Left in the Training Data :", train.isnull().sum().sum())

    # lets impute the missing values in the Testing Data
    test['education'] = test['education'].fillna(test['education'].mode()[0])
    test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].mode()[0])
    # lets check whether the Null values are still present or not?
    print("Number of Missing Values Left in the Training Data :", test.isnull().sum().sum())

    # Lets first analyze the Numberical Columns
    train.select_dtypes('number').head()

    # lets check the boxplots for the columns where we suspect for outliers
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.style.use('fivethirtyeight')
    # Box plot for average training score
    plt.subplot(1, 2, 1)
    sns.boxplot(train['avg_training_score'], color = 'red')
    plt.xlabel('Average Training Score', fontsize = 12)
    plt.ylabel('Range', fontsize = 12)
    # Box plot for length of service
    plt.subplot(1, 2, 2)
    sns.boxplot(train['length_of_service'], color = 'red')
    plt.xlabel('Length of Service', fontsize = 12)
    plt.ylabel('Range', fontsize = 12)
    plt.suptitle('Box Plot', fontsize = 20)
    plt.show()

    train = train[train['length_of_service'] > 13]

    # lets plot pie chart for the columns where we have very few categories
    plt.rcParams['figure.figsize'] = (16,5)
    plt.style.use('fivethirtyeight')
    # plotting a pie chart to represent share of Previous year Rating of the Employees
    """
    plt.subplot(1, 3, 1)
    labels = ['0','1']
    sizes = train['KPIs_met >80%'].value_counts()
    colors = plt.cm.Wistia(np.linspace(0, 1, 5))
    explode = [0, 0]

    plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
    plt.title('KPIs Met > 80%', fontsize = 20)
    """
    # plotting a pie chart to represent share of Previous year Rating of the Employees
    plt.subplot(1, 3, 2)
    labels = ['1', '2', '3', '4', '5']
    sizes = train['previous_year_rating'].value_counts()
    colors = plt.cm.Wistia(np.linspace(0, 1, 5))
    explode = [0, 0, 0, 0, 0.1]
    plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
    plt.title('Previous year Ratings', fontsize = 20)
    # plotting a pie chart to represent share of Previous year Rating of the Employees
    plt.subplot(1, 3, 3)
    labels = ['0', '1']
    sizes = train['awards_won'].value_counts()
    colors = plt.cm.Wistia(np.linspace(0, 1, 5))
    explode = [0,0.1]
    plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
    plt.title('Awards Won', fontsize = 20)
    plt.legend()
    plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    train_encoded = train.copy()
    label_encoder = LabelEncoder()
    for col in train_encoded.columns:
        if train_encoded[col].dtype == 'object':
            train_encoded[col] = label_encoder.fit_transform(train_encoded[col])
    plt.rcParams['figure.figsize'] = (15, 8)
    sns.heatmap(train_encoded.corr(), annot=True, linewidth=0.5, cmap='Wistia')
    plt.title('Correlation Heat Map', fontsize=15)
    plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (16, 7)
    sns.barplot(x=train['department'], y=train['avg_training_score'], hue=train['gender'], palette='autumn')
    plt.title('Chances of Promotion in each Department when they have won some Awards too', fontsize=15)
    plt.ylabel('Average Training Score', fontsize=10)
    plt.xlabel('Departments', fontsize=10)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

    # creating a Metric of Sum
    train['sum_metric'] = train['awards_won']+train['KPIs_met >80%'] + train['previous_year_rating']
    test['sum_metric'] = test['awards_won']+test['KPIs_met >80%'] + test['previous_year_rating']
    # creating a total score column
    train['total_score'] = train['avg_training_score'] * train['no_of_trainings']
    test['total_score'] = test['avg_training_score'] * test['no_of_trainings']

    train = train.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)
    test = test.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)
    # lets check the columns in train and test data set after feature engineering
    train.columns
    '''
    lets check the no. of employee who did not get an award, did not acheive 80+ KPI, previous_year_rating as 1
    and avg_training score is less than 40
    but, still got promotion.
    ''' 
    train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) & 
        (train['awards_won'] == 0) & (train['avg_training_score'] < 60) & (train['is_promoted'] == 1)]

    
    print("Before Deleting the above two rows :", train.shape)
    train = train.drop(train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) & 
        (train['awards_won'] == 0) & (train['avg_training_score'] < 60) & (train['is_promoted'] == 1)].index)
    # lets check the shape of the train data after deleting the two rows
    print("After Deletion of the above two rows :", train.shape)

    train.select_dtypes('object').head()

    train['education'].value_counts()

    train['education'] = train['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
    test['education'] = test['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                    (3, 2, 1))
    # lets use Label Encoding for Gender and Department to convert them into Numerical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train['department'] = le.fit_transform(train['department'])
    test['department'] = le.fit_transform(test['department'])
    train['gender'] = le.fit_transform(train['gender'])
    test['gender'] = le.fit_transform(test['gender'])
    # lets check whether we still have any categorical columns left after encoding
    print(train.select_dtypes('object').columns)
    print(test.select_dtypes('object').columns)

    train.head(3)

    y = train['is_promoted']
    x = train.drop(['is_promoted'], axis = 1)
    x_test = test
    print("Shape of the x :", x.shape)
    print("Shape of the y :", y.shape)
    print("Shape of the x Test :", x_test.shape)

    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    x_resample, y_resample = smote.fit_resample(x, y)
    print(x_resample.shape)
    print(y_resample.shape)

    print("Before Resampling:")
    print(y.value_counts())
    print("After Resampling:")
    print(y_resample.value_counts())

    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)
    # lets print the shapes again 
    print("Shape of the x Train :", x_train.shape)
    print("Shape of the y Train :", y_train.shape)
    print("Shape of the x Valid :", x_valid.shape)
    print("Shape of the y Valid :", y_valid.shape)
    print("Shape of the x Test :", x_test.shape)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_valid = sc.transform(x_valid)
    x_test = sc.transform(x_test)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)


    # Train DecisionTreeClassifier model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Save the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'model_decision_tree.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


    y_pred = model.predict(x_valid)

    #Accuracy Results
    from sklearn.metrics import confusion_matrix, classification_report
    TrainingAccuracy = "Training Accuracy :", model.score(x_train, y_train)
    TestingAccuracy = "Testing Accuracy :", model.score(x_valid, y_valid)
    cm = confusion_matrix(y_valid, y_pred)
    plt.rcParams['figure.figsize'] = (3, 3)
    sns.heatmap(cm, annot = True, cmap = 'Wistia', fmt = '.8g')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()

    # lets take a look at the Classification Report
    cr = classification_report(y_valid, y_pred)
    print(cr)

    return render(request,'users/machinelearning_result.html',{"TrainingAccuracy":TrainingAccuracy,"TestingAccuracy":TestingAccuracy,"heatmap_url": '/static/heatmap.png',"cm":cm,"cr":cr})

import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse

def prediction(request):
    if request.method == 'POST':
        try:
            # Collecting features from the request
            employee_id = request.POST.get("employee_id", "")
            department = request.POST.get("department", "")
            region = request.POST.get("region", "")
            education = request.POST.get("education", "")
            gender = request.POST.get("gender", "")
            recruitment_channel = request.POST.get("recruitment_channel", "")
            no_of_trainings = float(request.POST.get("no_of_trainings", 0))
            age = float(request.POST.get("age", 0))
            previous_year_rating = float(request.POST.get("previous_year_rating", 0))
            length_of_service = float(request.POST.get("length_of_service", 0))
            awards_won = float(request.POST.get("awards_won", 0))
            avg_training_score = float(request.POST.get("avg_training_score", 0))

            # Creating a DataFrame with the provided features (for future use)
            new_data = pd.DataFrame({
                'employee_id': [employee_id],
                'department': [department],
                'region': [region],
                'education': [education],
                'gender': [gender],
                'recruitment_channel': [recruitment_channel],
                'no_of_trainings': [no_of_trainings],
                'age': [age],
                'previous_year_rating': [previous_year_rating],
                'length_of_service': [length_of_service],
                'awards_won': [awards_won],
                'avg_training_score': [avg_training_score],
            })

            # Basic rule-based logic for promotion prediction
            if previous_year_rating >= 4 and avg_training_score > 80 and length_of_service > 5:
                msg = "Promoted"
            else:
                msg = "Not Promoted"

            return render(request, "users/predictForm.html", {"msg": msg})
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    else:
        return render(request, 'users/predictForm.html', {})

