from __future__ import with_statement
import os
import re
import pandas as pd
import pickle
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
import xlsxwriter 
GET_INV= True
dir_result="./result/"
ft_1=[]
ft_2=[]
ft_3=[]
ft_4=[]
ft_5=[]
ft_6=[]
ft_7=[]
labels=[]
fname=[]
utterances_count=[]
utterances=[]
def file_tokenization(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list = []
    for line in input_file:
        for element in line.split("\n"):
            if "*PAR" in element or ("*INV" in element and GET_INV):
                #remove any word after the period.
                cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR',''))
                #substitute numerical digits, deleting underscores
                cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                tokenized_list = word_tokenize(cleaned_string)
                output_list = output_list + tokenized_list
    return output_list
def generate_feature(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    f_1=f_2=f_3=f_4=f_5=f_6=f_7=count=0
    for line in input_file:
        for element in line.split("\n"):
            if "*PAR" in element or ("*INV" in element):
                
                #print(element)
                if '(.)' in element:
                    f_1+=element.count('(.)')
                elif '(..)' in element:
                    f_2+=element.count('(..)')
                elif '(...)' in element:
                    f_3+=element.count('(...)')
                elif '[/]' in element:
                    f_4+=element.count('[/]')
                elif '[//]' in element:
                    f_5+=element.count('[//]')
                elif '&uh' in element:
                    f_6+=element.count('&uh')
                elif '+...' in element:
                    f_7+=element.count('+...')
                count+=1
            
            
    ft_1.append(f_1)
    ft_2.append(f_2)
    ft_3.append(f_3)
    ft_4.append(f_4)
    ft_5.append(f_5)
    ft_6.append(f_6)
    ft_7.append(f_7) 
    utterances_count.append(count)    
    #return f_1,f_2,f_3,f_4,f_5,f_6,f_7
def file_tokenization_utterances(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list = []
    #count=0
    length=0
    previous="*INV"
    
    que_list = ["else",
                "anything", "anymore action", "can you","tell me", "mistakes", "how about", "what's going on over here",
                "going on",

                "is that all", "is that", "what's happening", " happening", "action", "some more", "more"]
    for line in input_file:
        for element in line.split("\n"):
            #if "*PAR" in element or ("*INV" in element and GET_INV):
            if "*INV" in element:
                
                #remove any word after the period.
                cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*INV',''))
                #substitute numerical digits, deleting underscores
                cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                if previous=="*PAR":
                    for k in range(len(que_list)):
                        if que_list[k] in cleaned_string:
                            count+=1
                            break
                tokenized_list = word_tokenize(cleaned_string)
                free_tokenized_list = []
                for element in tokenized_list:
                    if element is not '':
                        free_tokenized_list.append(element)
                output_list.append(free_tokenized_list)
            else:
                previous="*PAR"
    return output_list, count,length
def file_tokenization_both_utterances(input_file,count,filename):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list_inv = []
    output_list_par = []
    output_types=[]
    previous=None
   # print(count)
    length=0
    
    # que_list = ["else",
    #             "anything", "anymore action", "can you","tell me", "mistakes", "how about", "what's going on over here",
    #             "going on",
    #
    #             "is that all", "is that", "what's happening", " happening", "action", "some more", "more","?"]
    # neutral_list=["uhhuh","mhm","okay","alright"]
    utterance=0
    print("filename")
    print(filename)
    for line in input_file:
        for element in line.split("\n"):
            type=None
            tag=None
            if "*INV" in element or "*PAR" in element:
                # with open("./data/" + str(count) + '.txt', 'a') as f:
                    # cleaned_string = element.split('.', 1)[0]

                    # substitute numerical digits,
                    #
                    #
                    # deleting underscores

                    if len(element.split(':')[1])==1:


                        continue
                    else:
                        cleaned_string = re.sub(r'[\d]+', '', element.replace('_', ''))
                        utterances.append(cleaned_string)
                        fname.append(filename)
                        utterances_count.append(utterance)
                        utterance+=1
                    # f.write(cleaned_string + os.linesep)


               # if "*PAR" in element or ("*INV" in element and GET_INV):
                # cleaned_string = element.split('.', 1)[0]
                    # #replace par with empty string, deleting the part of the string that starts with PAR
                # if "*INV" in cleaned_string:
                    # #cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*INV',''))
                    # tag="*INV"
                # elif "*PAR" in cleaned_string:
                    # #cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*PAR',''))
                    # tag="*PAR"
                # #substitute numerical digits, deleting underscores
                # cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                # #tokenized_list = word_tokenize(cleaned_string)
                # #free_tokenized_list = []
                # # for element in tokenized_list:
                    # # if element is not '':
                        # # free_tokenized_list.append(element)
                # if tag=="*INV":
                    
                    # #remove any word after the period.
                    
                    # if (previous=="*INV"):
                        # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                            # f.write("*PAR: "+ os.linesep)
                        

                    # #output_list_inv.append(free_tokenized_list)
                    # # for k in range(len(neutral_list)):
                        # # if que_list[k] in cleaned_string:
                       
                            # # type="question"
                            # # break
                        
                    # # if type==None:
                        # # if "laughs" in cleaned_string:
                            # # type="imitate"
                        # # else:
                            # # for k in range(len(neutral_list)):
                                # # if neutral_list[k] in cleaned_string:
                                
                                    # # type="neutral"
                                    # # break
                        
                    
                    # previous="*INV"
                    
                    # #output_types.append(type)
                # elif tag =="*PAR" :
                    # #output_list_par.append(free_tokenized_list)
                    # if previous is "*PAR":
                       # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                            # f.write("*INV: "+ os.linesep)
                    # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                        # f.write(cleaned_string+ os.linesep)
                    # previous="*PAR"
                    # #output_types.append(type)
                
            
    #return output_list_inv, output_list_par,output_types
    #return length
def transcript_to_file():
    file_list=[16,17,20,21,22,26]
    PATH = "./data/script"
    count=51
    for path, dirs, files in os.walk(PATH):
        for filename in files:

            fullpath = os.path.join(path, filename)
            filename = int(filename.split('.')[0])
            if filename in file_list or filename>=37 and filename<=102:
                with open(fullpath, 'r', encoding="utf8",errors='ignore')as input_file:
                    file_tokenization_both_utterances(input_file,count,filename)
                    # count+=1
            else:
                continue
                # dementia_list.append(
                # {'text':tokenized_list,
                # 'label':label}
                # )
                # generate_feature(input_file)
                #
                # labels.append(label)
                # fname.append(filename)
                # break
                # dementia_list.append(
                # {'short_pause_count':f_1,
                # 'long_pause_count':f_2,
                # 'very_long_pause_count':f_3,
                # 'word_repetition_count':f_4,
                # 'retracing_count':f_5,
                # 'filled_pause_count':f_6,
                # 'incomplete_utterance_count':f_7,
                # 'label':label
                # }
                # )


def generate_full_interview_dataframe():
    """
    generates the pandas dataframe containing for each interview its label.
    :return: pandas dataframe.
    """
    dementia_list = []
    for label in ["Control", "Dementia"]:
        if label == "Dementia":
            folders = ["cookie"]
        else:
            folders = ["cookie"]

        for folder in folders:
            PATH = "./Dataset/DementiaBank/Pitt/Pitt/" + label + "/" + folder
            for path, dirs, files in os.walk(PATH):
                for filename in files:
                    fullpath = os.path.join(path, filename)
                    with open(fullpath, 'r',encoding="utf8")as input_file:
                        # tokenized_list = file_tokenization(input_file,label)
                        # dementia_list.append(
                            # {'text':tokenized_list,
                             # 'label':label}
                            # )
                        generate_feature(input_file)
                        
                        labels.append(label)
                        fname.append(filename)
                        #break
                        # dementia_list.append(
                            # {'short_pause_count':f_1,
                            # 'long_pause_count':f_2,
                            # 'very_long_pause_count':f_3,
                            # 'word_repetition_count':f_4,
                            # 'retracing_count':f_5,
                            # 'filled_pause_count':f_6,
                            # 'incomplete_utterance_count':f_7,
                            # 'label':label
                            # }
                            # )
                        
    #dementia_dataframe = pd.DataFrame(dementia_list)
    #return dementia_dataframe
def generate_single_utterances_dataframe():
    
    
    count=0
    
    PATH = "D:/Admission/UIC/conference/BHI/annotation/source" 
    for path, dirs, files in os.walk(PATH):
        for filename in files:
            fullpath = os.path.join(path, filename)
            with open(fullpath, 'r')as input_file:
                
                file_tokenization_both_utterances(input_file,filename)
                
               
                
                
                        

    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
# def generate_single_utterances_dataframe():
    
    # length=[]
    
    # ids=[]
    
    # count=0
    # for label in ["Control", "Dementia"]:
        # folders =  ["cookie"]
        # id = 0

        # for folder in folders:
            # PATH = "./Dataset/DementiaBank/Pitt/Pitt/" + label + "/" + folder
            # for path, dirs, files in os.walk(PATH):
                # for filename in files:
                    # fullpath = os.path.join(path, filename)
                    # with open(fullpath, 'r',encoding="utf8")as input_file:
                        # file_tokenization_both_utterances(input_file)
                        
                        # # dementia_list.append(
                        # # {
                        # # 'len':length,
                        # # 'label':label,
                        # # 'id':id
                        # # }
                        # #)
                        
                        # count+=1
                        # if count>25:
                            # break
                        # id = id +1
                        # ids.append(id)
                        # type.append(label)
                        
                # if count>25:
                    # count=1
                    # break    
                        # for element1,element2,element3 in zip(inv,par,type):
                            # dementia_list.append(
                            # {'par': element2,
                            # 'label': label,
                            # 'id':id,
                            # 'inv':element1,
                            # 'type':element3}
                                # )
                        

    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
transcript_to_file()
#dementia_dataframe=generate_full_interview_dataframe()
# generate_full_interview_dataframe()
# columns={'short_pause_count':ft_1,'long_pause_count':ft_2,'very_long_pause_count':ft_3,
#                             'word_repetition_count':ft_4,
#                             'retracing_count':ft_5,
#                             'filled_pause_count':ft_6,
#                             'incomplete_utterance_count':ft_7,
#                             'label':labels,
#                             'filename':fname,
#                             'count':utterances_count
#                             }
columns={'file':fname,'utterance_id':utterances_count,'utterances':utterances}
dementia_dataframe=pd.DataFrame(columns)
# # generate_single_utterances_dataframe()
#
# # dementia_dataframe = pd.DataFrame(dementia_list)
# # # print(dataframe)
# # # # with open('pitt_complete_dataframe_single_utterances.pickle', 'wb') as f:
#     # # # pickle.dump(dataframe, f)
writer = pd.ExcelWriter("./data/utterances.xlsx", engine='xlsxwriter')

# dementia_dataframe.to_excel(writer, sheet_name='Sheet1',columns=['short_pause_count','long_pause_count',
# 'very_long_pause_count','word_repetition_count','retracing_count','filled_pause_count','incomplete_utterance_count',
# 'label','filename','count'])
dementia_dataframe.to_excel(writer, sheet_name='Sheet1',columns=['file','utterance_id','utterances'])
writer.save()