# import xlsxwriter
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,jaccard_score,hamming_loss,average_precision_score,precision_recall_curve,precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
import os
import re
import xlrd
import seaborn as sns



real_value=[]
predicted_ovr=[]

id=[]



# Clean text

def preprocess(df):
    df["PAR"]=np.nan
    df["INV"] = np.nan

    for index, row in df.iterrows():

        # temp=row['utterance']
        # print()
        temp = df.iloc[index]['utterance']
        print(temp)
        #
        # print("index %d"%index)

        if(type(temp)==str):


            # cleaned_string = temp.split('.', 1)[0]
            cleaned_string = temp
            #replace par with empty string, deleting the part of the string that starts with PAR
            if "*INV" in cleaned_string:
                cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*INV',''))
                tag_inv=1
                tag_par = 0
            elif "*PAR" in cleaned_string:
                cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*PAR',''))
                tag_inv = 0
                tag_par = 1
            #substitute numerical digits, deleting underscores
            cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
            cleaned_string = cleaned_string.replace('[//]',"")
            cleaned_string = cleaned_string.replace('[/]', "")
            cleaned_string = cleaned_string.replace('(.)', "")
            cleaned_string = cleaned_string.replace('(..)', "")
            cleaned_string = cleaned_string.replace('(...)', "")

            print(cleaned_string)
            df.set_value(index, 'utterance', cleaned_string)

            # df.iloc[index]["utterance"]=df.iloc[index]["utterance"].replace(temp,cleaned_string,inplace=True)
            df.set_value(index, 'PAR', tag_par)
            df.set_value(index, 'INV', tag_inv)
            print(df.iloc[index]["file"])
            print(df.iloc[index]["utterance_id"])



    return df
#plot vovabulary feature
def plot_vocab(count_occur_df):
    g = count_occur_df.nlargest(columns="Count", n=50)
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=g, x="Count", y="Word")
    ax.set(ylabel='Count (appeared atleast 5 docs)')
    plt.show()
#print raw result
def print_result(Y_pred_ovr,Y_test,cols,count):

    for item in Y_pred_ovr:
        pred = []
        id.append(count)
        for val in range(22):

            try:
                if (float(item[val]) >= 0.5):
                    elem=str(cols[val])+str(item[val])
                    pred.append(elem)
            except ValueError:
                print("line %d in file %d" % (val, count))

        predicted_ovr.append(pred)


    pred.clear()

    for index,item in Y_test.iterrows():


        pred = []
        for val in range(22):

            try:
                if (float(item[cols[val]]) >= 0.5):
                    elem=str(cols[val])+str(item[cols[val]])
                    pred.append(elem)
            except ValueError:

                print("line %d in file %d" % (val, count))

        real_value.append(pred)
    pred.clear()
#Helper function to generate feature
def previous_utterance(df,index,n_label):
    print("prev")
    prev_utterance=[]

    data=df[df['file'].isin(index) ]
    print(data.iloc[0]['file'])

    for index,row in data.iterrows():

        if row['utterance_id']==0:
            prev_utterance.append([0]*n_label)
        else:
            if len(utterance_prev)==n_label:

                prev_utterance.append(utterance_prev)
        utterance_prev=[]
        for i in range(n_label):
           utterance_prev.append(row[i])




    return prev_utterance



 # Generate feature
def generate_ngrams(df,train_index,test_index,n_label):

    # print("testing..")
    # print(type(df))
    # vectorizer = CountVectorizer(min_df=10,ngram_range=(1, 3))


    # vectorizer = TfidfVectorizer(sublinear_tf=True,min_df=5, ngram_range=(1, 3),norm='l2')
    # train_text=df[df['file'].isin(train_index) ]["utterance"]
    # test_text=df[df['file'].isin(test_index) ]["utterance"]
    #
    # x_train = vectorizer.fit_transform(train_text)
    # y_train=df[df['file'].isin(train_index) ]
    # y_train=y_train.iloc[:,0:n_label]
    #
    #
    # x_test = vectorizer.transform(test_text)
    y_test = df[df['file'].isin(test_index)]
    y_test = y_test.iloc[:, 0:n_label]
    # vocab=vectorizer.get_feature_names()
    # count_list = x_train.toarray().sum(axis=0)
    # test_count_list=x_test.toarray().sum(axis=0)
    # ngram_feature=x_train.toarray()
    # # print(y_test.columns.values.tolist())
    # # print(y_test.shape)
    # # count_occur_df = pd.DataFrame(
    # #     (count, word) for word, count in
    # #     zip(x_train.toarray().tolist(),
    # #         vectorizer.get_feature_names()))
    # count_occur_df = pd.DataFrame({'Word':vocab,'Count':count_list})
    # count_occur_df_test = pd.DataFrame({'Word': vocab, 'Count': test_count_list})
    # # count_occur_df.columns = ['Word', 'Count']
    # count_occur_df.sort_values('Count', ascending=False, inplace=True)
    # count_occur_df_test.sort_values('Count', ascending=False, inplace=True)
    # print("train")
    # print(y_train)
    # print(x_train.shape[0])
    # print(x_train.shape[1])
    # print(x_test.shape[0])
    # print(x_test.shape[1])

    # print(count_list)
    # print("test")
    # print(y_train)
    # print(y_test)
    # dementia_dataframe = pd.DataFrame(columns)
    # df['ngram']=ngram_feature
    # print("ngram")
    # print(ngram_feature[0])
    # print("X")
    # print(x_train[0])
    # print("vocab")
    # # print(vocab[2])
    # print(len(vocab))
    # print(count_occur_df.head(20))
    # print(count_occur_df_test.head(20))
    # plot_vocab(count_occur_df)
    # return df,x_train,y_train,x_test,y_test
    return y_test

def plot_loss(model_scores):
    chain_len = 10
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    ax.set_title('Classifier Chain Ensemble Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation='vertical')
    ax.set_ylabel('Hamming loss')
    ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
    colors = ['r'] + ['b'] * chain_len + ['g']
    ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
    plt.tight_layout()
    plt.show()
    plt.savefig('hamming_tree.png')

#initiae and running classification model inthe method
def  Classifierchain_Model(X_train,Y_train,X_test,Y_test,cols,count):
    # Load a multi-label dataset from https://www.openml.org/d/40597



    # Fit an independent logistiec regression/decision tree etc model for each class using the
    # OneVsRestClassifier wrapper.
    # base_lr = LogisticRegression(solver='lbfgs')
    base_lr = DecisionTreeClassifier()
    # base_lr = LinearSVC(multi_class='ovr',max_iter=1000,tol=0.0001,penalty='l1', loss='squared_hinge', dual=False)
    ovr = OneVsRestClassifier(base_lr)
    ovr.fit(X_train, Y_train)
    Y_pred_ovr = ovr.predict(X_test)
    ovr_accuracy_score = accuracy_score(Y_test, Y_pred_ovr >= .5, normalize=True)
    ovr_jaccard_score =jaccard_score(Y_test, Y_pred_ovr>=.5, average='samples')

    ovr_recall_micro = recall_score(Y_test, Y_pred_ovr >= .5,
                                         average='micro')

    ovr_precision_micro = precision_score(Y_test, Y_pred_ovr >= .5, average='micro')

    ovr_f1_micro=f1_score(Y_test, Y_pred_ovr >= .5, average='micro')
    # baseline = DummyClassifier(strategy='stratified', random_state=0, constant=None)
    # baseline.fit(x_train, y_train)

    print_result(Y_pred_ovr,Y_test,cols,count)

    return ovr_accuracy_score,ovr_precision_micro,ovr_recall_micro,ovr_f1_micro,ovr_jaccard_score


def make_folds(train_index, test_index):
    train_set=[]
    test_set=[]
    df_fold = pd.read_excel('data/10_fold.xlsx')
    for index, row in df_fold.iterrows():
        for i in range(10):
            if index in train_index:
                train_set.append(row[i])
            else:
                test_set.append(row[i])

    print(train_set)
    print(test_set)
    return train_set,test_set
def write_result(name,data_frame,column_name):
    writer = pd.ExcelWriter("./data/"+name, engine='xlsxwriter')
    # #
    data_frame.to_excel(writer, sheet_name='Sheet1', column=column_name)
    writer.save()

def main():


    # File number in the data set, transcript id
    # X=[11,10,12,13,14,15,2,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,4,6,7,8,9,1]
    X=[1,2,3,4,5,6,7,8,9,0]
    # train_set=[]
    # test_set=[]
    utterance_test=[]
    utterance_file = []
    no_columns=0
    # Evaluation metric, for all the models in chain classifier
    model_average_accuracy_score = 0
    model_average_jaccard_score=0
    model_average_hamming_loss=0
    model_average_precision_weighted = 0
    model_average_precision_micro = 0
    model_average_recall_weighted =0
    model_average_recall_micro = 0
    model_average_f1_weighted = 0
    model_average_f1_micro = 0
    # number of iteration the model is run
    count=0
    # Reading dataset
    # with open('data/full_agreement_wide_extended.pickle', 'rb') as f:
    #     df_agreement = pickle.load(f)
    # df_agreement=preprocess(df_agreement)
    # with open('./data/full_agreement_wide_extended.pickle', 'wb') as f:
    #     pickle.dump(df_agreement, f)
    with open('data/full_agreement_wide_extended.pickle', 'rb') as f:
        df_wide1 = pickle.load(f)

    # Excluding the Label columns where the sum is 0, that label is not used

    no_columns=len(df_wide1.columns.values.tolist())-5
    with open('data/you_data.pickle', 'rb') as f:
        df_wide2 = pickle.load(f)

    # Excluding the Label columns where the sum is 0, that label is not used
    # df_agreement=df_wide.loc[:, (df_wide != 0).any(axis=0)]

    # print("columns")
    # print(len(df_agreement.columns.values.tolist())-5)
    #
    #
    #
    # # with open('data/full_agreement_wide.pickle', 'wb') as f:
    # #     pickle.dump(df_agreement, f)
    kf = KFold(n_splits=10)
    # # leave one out cross validation
    # loo = LeaveOneOut()
    # score=0
    #
    for train_index, test_index in kf.split(X):
        test_set=make_folds(train_index, test_index)

        y_test_real = df_wide1[df_wide1['file'].isin(test_set)]
        y_test_real = y_test_real.iloc[:, 0:no_columns]
        y_test_predicted = df_wide2[df_wide2['file'].isin(test_set)]
        y_test_predicted = y_test_predicted.iloc[:, 0:no_columns]
        # call the evaluation function

    #     test_utterance=df_agreement[df_agreement['file'].isin(test_set)]["utterance"]
    #     for item in test_utterance:
    #         utterance_test.append(item)
    #         utterance_file.append(test_set[0])
    #
    #     # generate feature of previous DA type
    #     prev_train=np.array(previous_utterance(df_agreement, train_set, no_columns))
    #     prev_test = np.array(previous_utterance(df_agreement, test_set, no_columns))
    #
    #
    #
    #     df,x_train,y_train,x_test,y_test=generate_ngrams(df_agreement,train_set, test_set,no_columns)
    #     train_feature_1=df[df['file'].isin(train_set) ]["PAR"]
    #     train_feature_2 = df[df['file'].isin(train_set)]["INV"]
    #     test_feature_1 = df[df['file'].isin(test_set)]["PAR"]
    #     test_feature_2 = df[df['file'].isin(test_set)]["INV"]
    #     train_feature_1=np.array(train_feature_1)
    #
    #
    #
    #     # adding speaker information as feature
    #
    #     x_train=np.concatenate((x_train.A, train_feature_1[:,None],train_feature_2[:,None]), axis=1)
    #     x_test = np.concatenate((x_test.A, test_feature_1[:,None],test_feature_2[:,None]), axis=1)
    #     # adding previous utterance label
    #     x_train=np.concatenate((x_train, prev_train), axis=1)
    #     x_test = np.concatenate((x_test, prev_test), axis=1)
    #
    # #
    # #     # x_train = np.concatenate((x_train.A, prev_train), axis=1)
    # #     # x_test = np.concatenate((x_test.A, prev_test), axis=1)
    #
    #
    # #
    #     # Creating model
    #     score,precision,recall,f1,jaccard=Classifierchain_Model(x_train, y_train, x_test, y_test, df_agreement.columns[0:no_columns], count)
    #
    #
    #
    #     #Accumulating result at each iteration
    #     model_average_accuracy_score+=score
    #     model_average_precision_micro+=precision
    #     model_average_recall_micro+=recall
    #     model_average_f1_micro+=f1
    #     model_average_jaccard_score+=jaccard
    # #
    # iteration count
        count+=1
    #     # break
    # print("result")
    #
    # print("accuracy, precision, recall, f1, jaccard %f %f %f %f %f "%(model_average_accuracy_score/count,model_average_precision_micro/count,model_average_recall_micro/count,model_average_f1_micro/count,model_average_jaccard_score/count))
    # # # Use this to write result
    # columns = {'id': id, 'Y_test': real_value, 'predicted_ovr': predicted_ovr}
    # result_df = pd.DataFrame(columns)
    # column_name=['id','Y_test','predicted_ovr']
    # name="raw_result_extended_tree.xlsx"
    # # write result to excel
    # write_result(name,result_df,column_name)







if __name__=="__main__":
    main()