# This project will use machine learning to classify a tumor as Benign(non-cancerous) or Malignant(Cancerous)
# The dataset used for this program can be accessed here: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# Users can use the input given by the dataset and submit it into the entry field of the GUI. 

# The code for this project has been classified into two parts 
# The first part will be dealing with building the GUI(graphical user interface) - using tkinter 
# The second part will deal with the implementing the computational aspect. 
# This program uses various python libraries to work 
# Inorder to access the libraries they must first be installed using the following commands: 
# pip install pillow 
# pip install numpy 
# pip install pandas
# pip install -U scikit-learn
# The program needs these libraries in order to work. 


####################################################################### PART 1 ############################################################################################
# Part One - Building the graphical user interface with tkinter 

# Importing the dependencies for tkinter 
import math
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

# Importing the dependencies for part two 
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# TKINTER - GUI
# building the tkinter window 
mainWindow = Tk()
mainWindow.geometry('1126x571')
mainWindow.rowconfigure(0, weight=1)
mainWindow.columnconfigure(0, weight=1)
mainWindow['background'] = '#00347A'
mainWindow.title("Tumor Classification Using Machine Learning")

# The UI will have 3 different windows
window1 = Frame(mainWindow)
window2 = Frame(mainWindow)
window3 = Frame(mainWindow)

for frame in (window1, window2, window3):
    frame.grid(row=0, column=0, sticky='nsew')
    frame.config(background='#00347A')


# The function show_frame() will display the page when called.
# For eg if window1 is given as an argument(show_frame(window1)) it will display window1
# It will start by displaying window1
def show_frame(frame):
    frame.tkraise()


show_frame(window1)

# The three windows have been built below starting with window1 

# ~~~~~~~~~~~~~~~~ window1 ~~~~~~~~~~~~~~~~
# Declaring the widgets of window1
load = Image.open('Images\\BackgroundImage.jpg')
render = ImageTk.PhotoImage(load)
img = Label(window1, image=render)
img.place(x=0, y=0)

label_1 = Label(window1, text='⟢ Machine Learning Project ⟢', bg='#00347A',fg='white',font=('Calibri', 25, 'bold'))
label_2 = Label(window1, text='Benign or Malignant', bg='#00347A',fg='white',font=('Calibri', 10, 'normal'))

label_3 = Label(window1, text='When a tumor forms inside a body it could be either', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_4 = Label(window1, text='benign or malignant tumor. Malignant tumors are cancerous', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_5 = Label(window1, text='and can spread around the rest of the body.Benign tumors', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_6 = Label(window1, text='are not cancerous and do not spread towards the rest of the body', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_7 = Label(window1, text='Our product will use machine learning to classify a tumor either as', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_8 = Label(window1, text='benign or malignant. The user will give the input data in the entry', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_9 = Label(window1, text='below. The program will then classify the tumor. The input data', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))
label_10 = Label(window1, text='can be taken from the kaggle data set', bg='#00347A',fg='white',font=('Calibri', 12, 'normal'))


get_input = Label(window1, text='Input data:',bg='#00347A',fg='white', font=('Calibri', 12, 'normal'))
get_input_entry = Entry(window1, highlightthickness=2, width=50)

calculate_button = Button(window1, text='Calculate',fg='#535353',font=('Calibri', 12, 'normal'), command=lambda: show_frame(window2))
calculate_button. config(width=8)

get_input_data_button = Button(window1, text='Get Input Data',fg='#535353',font=('Calibri', 12, 'normal'), command=lambda: show_frame(window3))
get_input_data_button.config(width=15)

# Placing the widgets of window1
label_1.place(x=650, y=120)
label_2.place(x=790, y=160)
label_3.place(x=650, y=220)
label_4.place(x=650, y=240)
label_5.place(x=650, y=260)
label_6.place(x=650, y=280)
label_7.place(x=650, y=300)
label_8.place(x=650, y=320)
label_9.place(x=650, y=340)
label_10.place(x=650, y=360)

get_input.place(x=650, y=395)
get_input_entry.place(x=730, y=395)
calculate_button.place(x=965, y=425)
get_input_data_button.place(x=832, y=425)
# ~~~~~~~~~~~~~~~~ end of window1 ~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~ window2 ~~~~~~~~~~~~~~~~
# Since the background image has already been loaded(in window1)
# we do not need to load it again
img = Label(window2, image=render)
img.place(x=0, y=0)

calculate_title = Label(window2, text='Calculate', bg='#00347A',fg='white',font=('Calibri', 25, 'bold'))
result_label = Label(window2, text='Result:', bg='#00347A',fg='white', font=('Calibri', 13, 'normal'))
result_text_box = Text(window2, height=8, width=35, wrap=WORD, highlightthickness=2)
result_text_box.config(highlightcolor='#DAF3F0')

back_button = Button(window2, text='Back',fg='#535353',font=('Calibri', 12, 'normal'), command=lambda: show_frame(window1))
back_button. config(width=8)

# Placing the widgets of window 2
calculate_title.place(x=850, y=120)
result_label.place(x=760, y=170)
result_text_box.place(x=760,y=200)
back_button.place(x=975, y=340)
# ~~~~~~~~~~~~~~~~ end of window2 ~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~ window3 ~~~~~~~~~~~~~~~~
img = Label(window3, image=render)
img.place(x=0, y=0)

window_3_title = Label(window3, text='Get Input Data', bg='#00347A',fg='white',font=('Calibri', 25, 'bold'))
window_3_label_a = Label(window3, text='The input data has 30 entries. Users can use the data', bg='#00347A',fg='white', font=('Calibri', 12, 'normal'))
window_3_label_b = Label(window3, text="below. This data is taken from kaggle dataset", bg='#00347A',fg='white', font=('Calibri', 12, 'normal'))

window_3_label_c = Label(window3, text='Sample Input Data:', bg='#00347A',fg='white', font=('Calibri', 12, 'normal'))
get_input_canvas = Canvas(window3, bg='#A7C7E7', width=350, height=200, bd=1)
get_input_1 = Label(window3, text="17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_2 = Label(window3, text="20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_3 = Label(window3, text="19.69,21.25,100,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_4 = Label(window3, text="11.42,20.08,77.58,386.1,0.1425,0.2839,0.2414,0.1052,0.2597,0.09744,0.4956,1.156,3.445,27.23,0.00911,0.07458,0.05661,0.01867,0.05963,0.009208,14.91,26.5,98.87,567.7,0.2098,0.8663,0.6869,0.2575,0.6638,0.173", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_5 = Label(window3, text="20.29,14.340135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575,0.1374,0.205,0.4,0.1625,0.2364,0.07678", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_6 = Label(window3, text="12.45,1507,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613,0.3345,0.8902,2.217,27.19,0.00751,0.03345,0.03672,0.01137,0.02165,0.005082,15.47,23.75,103.4,741.6,0.1791,0.5249,0.5355,0.1741,0.3985,0.1244", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_7 = Label(window3, text="18.25,19.98,019.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_8 = Label(window3, text="13.71,20.83,00.2,577.9,0.1189,0.1645,0.09366,0.05985,0.2196,0.07451,0.5835,1.377,3.856,50.96,0.008805,0.03029,0.02488,0.01448,0.01486,0.005412,17.06,28.14,110.6,897,0.1654,0.3682,0.2678,0.1556,0.3196,0.1151", bg="#A7C7E7", fg="black", font="Calibri 10")
get_input_9 = Label(window3, text="16.02,23.24,102.7,797.8,0.08206,0.06669,0.03299,0.03323,0.1528,0.05697,0.3795,1.187,2.466,40.51,0.004029,0.009269,0.01101,0.007591,0.0146,0.003042,19.19,33.88,123.8,1150,0.1181,0.1551,0.1459,0.09975,0.2948,0.08452", bg="#A7C7E7", fg="black", font="Calibri 10")

# Placing the widgets of window 3
window_3_title.place(x=800, y=120)
window_3_label_a.place(x=740, y=180)
window_3_label_b.place(x=740, y=200)
window_3_label_c.place(x=740, y=225)
get_input_canvas.place(x=740, y=245)
get_input_1.place(x=740, y=250)
get_input_2.place(x=740, y=270)
get_input_3.place(x=740, y=290)
get_input_4.place(x=740, y=310)
get_input_5.place(x=740, y=330)
get_input_6.place(x=740, y=350)
get_input_7.place(x=740, y=370)
get_input_8.place(x=740, y=390)
get_input_9.place(x=740, y=410)
# ~~~~~~~~~~~~~~~~ end of window3 ~~~~~~~~~~~~~~~~



####################################################################### PART 2 ############################################################################################
# Part Two - Computational Implementation 
def calculatePrediction(): 
    # 2.1) Data Collection & Processing
    # loading the data from sklearn
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

    # To view the dataset we can use the command below 
    # print(breast_cancer_dataset)

    # loading the data to a data frame
    # The data_frame helps us organize the dataset into a tabular format - making it more readable 
    # The contents of the data_frame are the features(as the column) and the coressponding data(as the rows)
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

    # To view the first 5 rows of the dataframe we can use the command below 
    # data_frame.head()

    # adding the 'target' column to the data frame which will contain either 0(Malignant) or 1(Benign)
    data_frame['label'] = breast_cancer_dataset.target

    # To print the last 5 rows of the dataframe we can use this commad 
    # data_frame.tail()

    # To view the number of rows and columns in the dataset
    # data_frame.shape    #(569, 31)

    # To get more information about the data
    # such as the datatype each feature contains and wether or not null values are permissible 
    data_frame.info()

    # To check the number of null values that have been entered for the features can also checkfor missing values
    data_frame.isnull().sum()

    # To get statistical measures about the data. For example: the mean value of the features, as well as the minimum and maximum value
    data_frame.describe()

    # To check the distribution of target varibales - how many are benign(non-cancerous) and how many are malignant(cancerous)
    data_frame['label'].value_counts()

    data_frame.groupby('label').mean()

    #####################################################
    # 2.2) Separating the features from the target 
    # X - represents the features only since the label/target value has been dropped. 
    # Y - represents the label or the traget without the features. 
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']

    # To view the value od X and Y 
    # print(X)
    # print(Y)

    #####################################################
    # 2.3) Splitting the data into training data & Testing data
    # 20% of the data will be testing data while 80% will be training data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # print(X.shape, X_train.shape, X_test.shape)

    #####################################################
    # 2.4) Model Training
    # Logistic Regression 
    # getting an instance from logistic regression and storing it in identifier named model 
    # train the model usinf fit()
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    #####################################################
    # 2.5) Model Evaluation
    # Accuracy Score 

    # To check the accuracy on training data we have used accuracy_score()
    # which compares the models prediction(X_train_prediction) with the actual value/prediction(Y_train)
    # The accuracy of the model on the training data is 0.945 which can be approximated to 95% accurate 
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction) 

    # print('Accuracy on training data = ', training_data_accuracy)

    # To check accuracy on test data 
    # The accuracy of the model on the testing data is 0.935 which can be approximated to 94% accurate 
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction) 

    # print('Accuracy on test data = ', test_data_accuracy)

    #######################
    # 2.5) Finalization 
    # recieving the data from the user 
    # However the data is orginally recieved as string but it must be convereted to numpy array
    # To do so the string value(input_data_as_string) must be convereted to tuple. 
    # It is convereted then stored in the variable named as res 
    # The tuple(res) is then converted to numpy array by using the np.asarray() function

    input_data_as_string= get_input_entry.get()
    res = tuple(map(float, input_data_as_string.split(',')))
    print(res)
    input_data_as_numpy_array = np.asarray(res)


    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        result_message_4 = "The Breast cancer is Malignant"
        result_text_box.insert('end', result_message_4)
      

    else:
        result_message_5 = "The Breast cancer is Benign"
        result_text_box.insert('end', result_message_5)


# TKINTER BUTTONS
# The reason the button has been implemented here is because it calls the function calculatePrediction()
# However we need to write these functions before implementing the buttons otherwise the program will not know them.
generate_result_button = Button(window2, text='Get Result',fg='#535353',font=('Calibri', 12, 'normal'), command=calculatePrediction)
generate_result_button.config(width=9)
generate_result_button.place(x=900, y=340)
mainWindow.mainloop()

# Note: After running the program the user can take sample input data from the dataset which can be found in the dataset folder.
# When taking sample input data only the 30 features are considered so no need to put the target(diagnosis) or the id 
# The input data has 30 entries since there are 30 features each of the entries must be separted with a comma
# Or user can copy and paste the following sample input data: 
# 13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
# 17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
# 20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902
# 19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758
# 11.42,20.38,77.58,386.1,0.1425,0.2839,0.2414,0.1052,0.2597,0.09744,0.4956,1.156,3.445,27.23,0.00911,0.07458,0.05661,0.01867,0.05963,0.009208,14.91,26.5,98.87,567.7,0.2098,0.8663,0.6869,0.2575,0.6638,0.173
