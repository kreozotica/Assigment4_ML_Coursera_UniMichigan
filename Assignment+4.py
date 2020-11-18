
# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary
# Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one
# of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs)
# are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city
# of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid
# blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv.
# Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom
# each ticket was issued.
# The target variable is compliance, which is
# True if the ticket was paid early, on time, or within one month of the hearing data,
# False if the ticket was paid after the hearing date or not at all, and
# Null if the violator was found not responsible.
# Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation.
# They are included in the training set as an additional source of data for visualization, and to enable
# unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit
# using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability
# that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
#
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question.
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.

# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket.
# This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket
# will be paid on time.
#



def blight_model():


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    import traceback
    import string
    from sklearn.base import BaseEstimator
    from category_encoders.ordinal import OrdinalEncoder
    import category_encoders.utils as util
    from sklearn.utils.random import check_random_state
    from feature_engine import categorical_encoders as ce
    import xgboost as xgb
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from xgboost import plot_importance
    import xgboost
    from matplotlib import pyplot
    import category_encoders as ces
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.kernel_approximation import RBFSampler
    from xgboost import XGBClassifier

    from category_encoders.cat_boost import CatBoostEncoder

    from sklearn.metrics import confusion_matrix, roc_curve, auc, plot_roc_curve, accuracy_score
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score


    train = pd.read_csv('train.csv', encoding = 'ISO-8859-1')
    test = pd.read_csv('test.csv')



    #train_no_null['compliance_detail'].unique()
    #a =train_no_null[train_no_null['compliance_detail'] == 'non-compliant by no payment']
    #a['payment_status'].unique()
    #a['compliance'].unique()

    ############################################################
    ########## DATA CLEARNING & DATA LEAKAGE PREVENT ###########
    ############################################################

    train_no_null = train.loc[train.compliance.notnull()]


    ## ifdentifying indecies which are not satisfy conditions
    badValuesTrain = []
    for index, row in train_no_null.iterrows():
        if (train_no_null['payment_status'].loc[index] == 'PAID IN FULL') and (train_no_null['compliance'].loc[index] == 0)  and  (train_no_null['compliance_detail'].loc[index] == 'non-compliant by late payment more than 1 month') and (train_no_null['compliance'].loc[index] == 1)\
                or (train_no_null['payment_status'].loc[index] == 'NO PAYMENT APPLIED') and (train_no_null['compliance'].loc[index] == 1)\
                or (train_no_null['payment_status'].loc[index] == 'PARTIAL PAYMENT APPLIED') and (train_no_null['compliance'].loc[index] == 1)\
                or (train_no_null['payment_status'].loc[index] == 'NO PAYMENT APPLIED') and (train_no_null['compliance_detail'].loc[index] == 'compliant by no fine') and (train_no_null['compliance'].loc[index] == 1):
            badValuesTrain.append(index)


    # remove obtained indexes from the initial DF using QUERY
    a = train_no_null.query('index not in @badValuesTrain')

    # how many NaNs per column in TRAIN DATA

    train_no_null = train_no_null.query("state == state")
    train_no_null = train_no_null.query("zip_code == zip_code")
    train_no_null = train_no_null.query("mailing_address_str_number == mailing_address_str_number")
    train_no_null = train_no_null.query("mailing_address_str_name == mailing_address_str_name")

    #test = test.query("state == state")
    #test = test.query("zip_code == zip_code")
    #test = test.query("city == city")
    #test = test.query("violator_name == violator_name")
    #test = test.query("mailing_address_str_number == mailing_address_str_number")
    #test = test.query("mailing_address_str_name == mailing_address_str_name")


    #train_no_null.isnull().sum(axis = 0)
    #test.isnull().sum(axis = 0)

    train_no_null['hearing_date'].fillna(train_no_null['hearing_date'].value_counts().index[0], inplace=True)
    test['hearing_date'].fillna(test['hearing_date'].value_counts().index[0], inplace=True)
    test['state'].fillna(test['state'].value_counts().index[0], inplace=True)
    test['zip_code'].fillna(test['zip_code'].value_counts().index[0], inplace=True)
    test['mailing_address_str_number'].fillna(test['mailing_address_str_number'].value_counts().index[0], inplace=True)
    test['mailing_address_str_name'].fillna(test['mailing_address_str_name'].value_counts().index[0], inplace=True)


    # remove the colums from TRAINING data which are not corresponds to TEST data
    # getting a list of common columns betwee TRAIN and TEST
    common_cols = list(set(train_no_null.columns).intersection(test.columns))
    train_upd = train_no_null[common_cols]
    removedColumnsTrain = train_no_null.drop([col for col in train_no_null.columns if col in train_no_null.columns and col in test.columns], axis=1)
    y_train = removedColumnsTrain['compliance']

    # remove colums with lots of NaNs for both TRAIN and TEST DS
    train_upd = train_upd.drop(['non_us_str_code'], axis=1)
    test = test.drop(['non_us_str_code'], axis=1)
    train_upd = train_upd.drop(['violation_zip_code'], axis=1)
    test = test.drop(['violation_zip_code'], axis=1)
    train_upd = train_upd.drop(['grafitti_status'], axis=1)
    test = test.drop(['grafitti_status'], axis=1)



    #####################################################################
    ##################### PLOTTING/CLEANING #############################
    #####################################################################
    #train_upd.plot(subplots=True, layout=(4,3))
    #test.plot(subplots=True, layout=(4,3))
    #plt.close('figure')
    # since "state_fee", "clean_up_cost", "admin_fee" have no impact factor, they are constant, we remove them
    train_upd = train_upd.drop(['state_fee'], axis=1)
    test = test.drop(['state_fee'], axis=1)
    train_upd = train_upd.drop(['clean_up_cost'], axis=1)
    test = test.drop(['clean_up_cost'], axis=1)
    train_upd = train_upd.drop(['admin_fee'], axis=1)
    test = test.drop(['admin_fee'], axis=1)

    ################# EXTRA PLOTING FEATURES ###############################

    def plot_Comp_train_test(train, test, plotVar, titleName, plotShowNumsorted=30, plotkind='bar', figsize=(18, 3.2)):
        plt.subplots(1, 2, figsize=(18, 5))

        plt.subplot(1, 2, 1)
        yvalue = train[plotVar].value_counts()
        (yvalue[:plotShowNumsorted] / train.shape[0]).plot(kind="bar", alpha=0.6, color='slateblue')
        plt.title(titleName + ' (training set)')

        plt.subplot(1, 2, 2)
        yvalue = test[plotVar].value_counts()
        (yvalue[:plotShowNumsorted] / test.shape[0]).plot(kind="bar", alpha=0.6, color='teal')
        plt.title(titleName + ' (test set)')

        return plt


# plot_Comp_train_test(train_upd, test, 'zip_code', 'zip_code', plotShowNumsorted=55, figsize=(20,3.2));
# plot_Comp_train_test(train_upd, test, 'violation_code', 'violation_code', plotShowNumsorted=55, figsize=(20,3.2));

##################################################################
############# FEATURES PREPROCESSING  REGEX  #####################
##################################################################
##################################################################

################# CREATING DATE & TIME FEATURES ##################
##################################################################


    train_upd['ticket_issued_date'] = pd.to_datetime(train_upd.ticket_issued_date, format='%Y-%m-%d %H:%M:%S')
    train_upd['hearing_date'] = pd.to_datetime(train_upd.hearing_date, format='%Y-%m-%d %H:%M:%S')
    test['ticket_issued_date'] = pd.to_datetime(test.ticket_issued_date, format='%Y-%m-%d %H:%M:%S')
    test['hearing_date'] = pd.to_datetime(test.hearing_date, format='%Y-%m-%d %H:%M:%S')

    datetime = ['day', 'month', 'year', 'hour', 'minute', 'weekday', 'week']

    for period in datetime:
        if datetime != 'week':
            train_upd['Issued_' + period] = getattr(train_upd.ticket_issued_date.dt, period)
            test['Issued_' + period] = getattr(test.ticket_issued_date.dt, period)
            train_upd['Hearing_' + period] = getattr(train_upd.hearing_date.dt, period)
            test['Hearing_' + period] = getattr(test.hearing_date.dt, period)
        else:
            train_upd['Issued_' + period] = getattr(train_upd.ticket_issued_date.dt.isocalendar(), period)
            test['Issued_' + period] = getattr(test.ticket_issued_date.dt.isocalendar(), period)
            train_upd['Hearing_' + period] = getattr(train_upd.hearing_date.dt.isocalendar(), period)
            test['Hearing_' + period] = getattr(test.hearing_date.dt.isocalendar(), period)

    # removing columns with DataTime
    train_upd = train_upd.drop(['ticket_issued_date'], axis=1)
    train_upd = train_upd.drop(['hearing_date'], axis=1)
    test = test.drop(['ticket_issued_date'], axis=1)
    test = test.drop(['hearing_date'], axis=1)

    #train_upd.isnull().sum(axis=0)

    ### cleaning mailing_address_str_number column ####
    for i, row in list(test.iterrows()):

        if type(row['mailing_address_str_number']) != 'int':
            c = str(row['mailing_address_str_number'])


        if ('p' in row['mailing_address_str_number'].lower()) or ('*' in row['mailing_address_str_number']) \
                or ('.' in row['mailing_address_str_number']) or ('O' in row['mailing_address_str_number']) \
                or ('o' in row['mailing_address_str_number']) or ('G' in row['mailing_address_str_number']) \
                or ('# 143' in row['mailing_address_str_number'] )  or ('XX' in row['mailing_address_str_number']) \
                or ('22A' in row['mailing_address_str_number']) or ('NE' in row['mailing_address_str_number'])\
                or ('12 1ST' in row['mailing_address_str_number']) or ('11111A' in row['mailing_address_str_number']) :
            test.at[i,'mailing_address_str_number'] = 11111
            #print(i, test.at[i,'mailing_address_str_number'])

    test.mailing_address_str_number = test.mailing_address_str_number.replace(' ','',regex=True).replace(',','',regex=True)
    test.mailing_address_str_number = test.mailing_address_str_number.replace(to_replace='[A-Z-a-z][0-9]*', value = '11111', regex=True).replace('-','',regex=True).replace('`','',regex=True).replace('#','11111',regex=True)



### converting mailing adrees for both TRAIN and TEXT into numbers instread of stirngs
    for i, row  in list(train_upd.iterrows()):
        if isinstance(train_upd.at[i,'mailing_address_str_number'], float) == False:
            train_upd.at[i,'mailing_address_str_number'] = float(train_upd.at[i,'mailing_address_str_number'])

    for i, row  in list(test.iterrows()):
        if isinstance(test.at[i,'mailing_address_str_number'], float) == False:
            test.at[i,'mailing_address_str_number'] = float(test.at[i,'mailing_address_str_number'])



    ######### categorical encoding Weight of Evidence #########
    ###########################################################
    ####### Weight of Evidence transformation of text values into categories ##############
    cat_columns = ['country', 'city', 'state', 'agency_name', 'disposition', 'zip_code', 'mailing_address_str_name',  'violator_name', 'violation_street_name', 'violation_code',
                   'violation_description', 'inspector_name']


    woe_encoder = ces.WOEEncoder(cols=cat_columns)
    #fit the encoder
    woe_encoded_train = woe_encoder.fit_transform(train_upd.iloc[:], y_train)
    woe_encoded_train = woe_encoder.fit_transform(train_upd, y_train)
    # transform
    XTrain_transformed = woe_encoder.transform(train_upd)
    XTest_transformed = woe_encoder.transform(test)


   # CBE_encoder = CatBoostEncoder()
   # train_encoded = CBE_encoder.fit_transform(train_upd[cat_columns], y_train)
   # test_encoded = CBE_encoder.transform(test[cat_columns])
   # t = train_upd
   # t = t.drop(['country', 'city', 'state', 'agency_name', 'disposition', 'zip_code', 'mailing_address_str_name', 'violator_name', 'violation_street_name', 'violation_code', 'violation_description', 'inspector_name'], axis=1, inplace=True)
   # tt = test
   # tt = tt.drop(['country', 'city', 'state', 'agency_name', 'disposition', 'zip_code', 'mailing_address_str_name', 'violator_name', 'violation_street_name', 'violation_code', 'violation_description', 'inspector_name'], axis=1, inplace=True)

   # XTrain_transformed = pd.concat([train_upd, train_encoded], axis=1, sort=False)
   # XTest_transformed = pd.concat([test, test_encoded], axis=1, sort=False)


##########################################################
############# Correlation map for features ###############
##########################################################

    correlation = XTrain_transformed.corr().round(1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    sns.heatmap(data=correlation, annot=True, cmap="YlGn")
    ax.set_title("Correlation matrix for taken variables");

    #plt.savefig('plots/correlationMap.pdf')
    ##########################################################
    ################## saving new data #######################
    ##########################################################

    #XTrain_transformed.to_csv(r'/Users/kreozotica/PycharmProjects/current/ML_Coursera/processed_train.csv', index=False)
    #XTest_transformed.to_csv(r'/Users/kreozotica/PycharmProjects/current/ML_Coursera/XTest_transformed.csv', index=False)


    #############################################################################################################################
    # Further, since we don't have prediction data, we keep TEST data as prediction and SPLIT TRAIN data into new TRAIN and TEST
    #############################################################################################################################

    X_train, X_test, y_train, y_test = train_test_split(XTrain_transformed, y_train, random_state=0, test_size=0.75)
    #### scalling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    XTest_transformed_scaled = scaler.fit_transform(XTest_transformed)
    XTest_transformed_scaled = pd.DataFrame(XTest_transformed_scaled, columns = XTest_transformed.columns)


    #plot_importance(model_XGB)
    #pyplot.show()
    # featureImportance = pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=TrainTest_noLabel.columns)


    ##########################################################
    ################## MODELLING APPROACH ####################
    ##########################################################

    ##### universal model's function
    def modelFit(X_train, X_test, y_train, y_test, clf, cv=5):

        clf = clf.fit(X_train, y_train)

        cv = cross_val_score(clf, X_test, y_test, cv=cv, scoring = 'roc_auc')
        cv_mean = round(cv.mean(), 3)
        cv_std = round(cv.std(), 3)
        print('Cross-validation (AUC)', cv, ', mean =', cv_mean, ', std =', cv_std)

        #y_pred =clf.predict(X_test)
        #confusion = confusion_matrix(y_test, y_pred)
        #print(confusion)

        return cv_mean, cv_std

    ##### XGBoost
    clf_XGB = XGBClassifier()
    auc_mean_XGB, auc_std_XGB = modelFit(X_train_scaled, X_test_scaled, y_train, y_test, clf_XGB, cv=20)


    ##### Gradient-boosted Decision Trees¶

    clf_GBC = GradientBoostingClassifier(learning_rate=0.05)
    # scaling doesn't really need it, advantage
    auc_mean_GBC, auc_std_GBC = modelFit(X_train_scaled, X_test_scaled, y_train, y_test, clf_GBC, cv=20)

    ##### SVM

    clf_SVM = SVC(kernel='rbf', C=1, random_state=0)
    auc_mean_SVM, auc_std_SVM = modelFit(X_train_scaled, X_test_scaled, y_train, y_test, clf_SVM, cv=20)

    #### LogReg

    grid_values = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    LogReg = LogisticRegression()
    grid_rbf_recall = GridSearchCV(LogReg, param_grid = grid_values, scoring='recall')
    auc_mean_LR, auc_std_LR = modelFit(X_train_scaled, X_test_scaled, y_train, y_test, grid_rbf_recall, cv=20)

    #### RidgeReg
    #RdgReg_clf = Ridge()
    #auc_mean_RG, auc_std_RG = modelFit(X_train_scaled, X_test_scaled, y_train, y_test, RdgReg_clf, cv=20)

    ### NaiveBayes

    NB_clf = GaussianNB()
    auc_mean_NB, auc_std_NB = modelFit(X_train_scaled, X_test_scaled, y_train, y_test, NB_clf, cv=20)

    ################## ROC vis ##################
    def roCurves(clfList, X_test, y_test):

        roCurveList = []
        plt.subplots(1, 1, figsize=(5, 5))
        styleList = ['solid', 'solid', 'dashed', 'dashed', 'dotted', 'dashed']

        for clf, sty in zip(clfList, styleList):
            ax = plt.gca()
            roc = plot_roc_curve(clf, X_test, y_test, ax=ax, alpha=0.85, lw=2, linestyle=sty)
            roCurveList.append(roc)
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='dotted')
        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        return roCurveList


    exps = [clf_XGB, clf_GBC, clf_SVM, grid_rbf_recall, NB_clf]

    roCurves(exps, X_test_scaled, y_test)

    # Save the figure and show
    #plt.tight_layout()
    #plt.savefig('plots/ROCs.png')
    #plt.show()



    ##### Pedict probabilities for the best model - XGBoost
    y_proba = clf_XGB.predict_proba(XTest_transformed_scaled)[:,1]
    # Integrate with reloaded test data
    test['compliance'] = y_proba




    return  test.compliance

blight_model()
