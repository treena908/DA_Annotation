import pickle
from nltk.metrics.agreement import AnnotationTask
# from nltk.metrics import agreement
from pandas import ExcelWriter
from pandas import ExcelFile
import pandas as pd
import json
import os
file_id=[]
utterance_id=[]
label=[]
list_shahla=[]
list_mina=[]

exclude=0
# sum=0


def convert(s,ch): 
  
    
    return [i for i, ltr in enumerate(s) if ltr == ch]
def preprocess(row,name):
    # if(type(row)==float):
    #     return
    start=0
    str_shahla=row[1:-1]

    
    index_shahla=convert(str_shahla,',')
    #print(index_shahla)
    
    if len(index_shahla)>0:
        
        for i in index_shahla:
            if name=='s':
                list_shahla.append(str_shahla[start:i][1:-1])
            else:
                list_mina.append(str_shahla[start:i][1:-1])
            start=i+2
        if name=='s':
                list_shahla.append(str_shahla[start:len(str_shahla)][1:-1])
        else:
            list_mina.append(str_shahla[start:len(str_shahla)][1:-1])
        # print(list)
        #print(list_shahla)
        #return list_shahla
    else:
        #print(str_shahla[1:-1])
        if name=='s':
            list_shahla.append(str_shahla[1:-1])
        else:
            list_mina.append(str_shahla[1:-1])
        #print(list_shahla.append(str_shahla[1:-1]))
        
        #print(str_shahla[1:-1])
        #return list_shahla.append(str_shahla[1:-1])
def check():
    list=['Answer: t3','Answer: t2','Answer: t1','Answer: t4','Answer: t5','Answer: t6','Answer: t7','Answer: t8']
    count=0
    print(list_mina)
    for item in list_mina:
        if item not in list:
            
            count+=1
            
            #print(count)
            return count
    return count
def make_wide_dataframe(df_agreement):
    dic_dist = {
        'Answer:t3': [0]*1616,
        'Answer:t2': [0]*1616,
        'Answer:t1': [0]*1616,
        'Answer:t4': [0]*1616,
        'Answer:t5':[0]*1616,
        'Answer:t6': [0]*1616,
        'Answer:t7': [0]*1616,
        'Answer:t8': [0]*1616,
        'Question:General': [0]*1616,
        'Question:Reflexive': [0]*1616,
        "Answer:Yes": [0]*1616,
        "Answer:No": [0]*1616,
        "Answer:General": [0]*1616,
        "Instruction": [0]*1616,
        "Suggestion": [0]*1616,
        "Request": [0]*1616,
        "Offer": [0]*1616,
        "Acknowledgment": [0]*1616,
        "Request:Clarification": [0]*1616,
        "Feedback:Reflexive": [0]*1616,
        "Stalling": [0]*1616,
        "Correction": [0]*1616,
        "Farewell":[0]*1616,
        "Apology": [0]*1616,

        "Other": [0]*1616

    }
    track=0
    flag=0
    for index, row in df_agreement.iterrows():
        try:
            # print("index %d" % (index))
            # print(row)
            if pd.isnull(row['curated']) and pd.isnull(row['shahla']):
                break
            elif pd.isnull(row['curated']):
                preprocess(row['shahla'], 's')
            else:
                preprocess(row['curated'], 's')
            # print(list_shahla)
            for item in list_shahla:
                print("index")
                dic_dist[item.replace(" ", "")][index]=1
                if flag==0:
                    flag=1
                    track += 1
                    print(row['file'])
                    print(row['utterance_id'])
            flag=0

            list_shahla.clear()

        except KeyError:

            list_shahla.clear()
            continue
    df_wide = pd.DataFrame.from_dict(dic_dist )
    df_wide['file']=df_agreement.iloc[0:1616]['file']
    df_wide['utterance_id'] = df_agreement.iloc[0:1616]['utterance_id']
    df_wide['utterance'] = df_agreement.iloc[0:1616]['utterance']
    print(len(df_wide))
    print(track)
    return df_wide


def distribution(df_agreement):

    dic_dist = {
   'Answer:t3': {'*INV':0,'*PAR':0,'total':0},
   'Answer:t2':{'*INV':0,'*PAR':0,'total':0},
    'Answer:t1': {'*INV':0,'*PAR':0,'total':0},
    'Answer:t4': {'*INV':0,'*PAR':0,'total':0},
    'Answer:t5': {'*INV':0,'*PAR':0,'total':0},
    'Answer:t6': {'*INV':0,'*PAR':0,'total':0},
    'Answer:t7': {'*INV':0,'*PAR':0,'total':0},
    'Answer:t8': {'*INV':0,'*PAR':0,'total':0},
    'Question:General': {'*INV':0,'*PAR':0,'total':0},
    'Question:Reflexive':{'*INV':0,'*PAR':0,'total':0},
    "Answer:Yes": {'*INV':0,'*PAR':0,'total':0},
    "Answer:No": {'*INV':0,'*PAR':0,'total':0},
    "Answer:General": {'*INV':0,'*PAR':0,'total':0},
    "Instruction": {'*INV':0,'*PAR':0,'total':0},
    "Suggestion": {'*INV':0,'*PAR':0,'total':0},
    "Request": {'*INV':0,'*PAR':0,'total':0},
    "Offer": {'*INV':0,'*PAR':0,'total':0},
    "Acknowledgment":{'*INV':0,'*PAR':0,'total':0},
    "Request:Clarification":{'*INV':0,'*PAR':0,'total':0},
    "Feedback:Reflexive": {'*INV':0,'*PAR':0,'total':0},
    "Stalling": {'*INV':0,'*PAR':0,'total':0},
    "Correction": {'*INV':0,'*PAR':0,'total':0},
    "Farewell": {'*INV':0,'*PAR':0,'total':0},
    "Apology": {'*INV':0,'*PAR':0,'total':0},
    "Greeting": {'*INV':0,'*PAR':0,'total':0},
    "Other": {'*INV':0,'*PAR':0,'total':0}

    }
    list=['Answer:t3','Answer:t2','Answer:t1','Answer:t4','Answer:t5','Answer:t6','Answer:t7','Answer:t8']
    count=0
    track=0
    list_len=0
    speaker=0
    for index, row in df_agreement.iterrows():
        if '*PAR' in row['utterance'] or '*INV' in row['utterance']:
            speaker+=1

        try:
            # print(type(row['utterance']) )
            # print(row['file'])
            # print(row['utterance_id'])



            if type(row['utterance'])==str:
                # print("inside")

                # print(row['utterance'])
                if pd.isnull(row['curated']):
                    # print(index)
                    # print(type(row['shahla']))

                    preprocess(row['shahla'],'s')
                    count += 1
                else:

                    preprocess(row['curated'],'s')
                    count += 1
                # print("list")
                # print(list_shahla)

                if len(list_shahla)>0:

                    list_len+=1
                else:

                    print("no list")
                    print(row['file'])
                    print(row['utterance_id'])
                for item in list_shahla:



                    if '*PAR'==row['utterance'].split(':')[0]:



                        dic_dist[item.replace(" ", "")]['*PAR']+=1


                    elif '*INV'==row['utterance'].split(':')[0]:


                        dic_dist[item.replace(" ", "")]['*INV']+=1
                    track += 1





                    break

                list_shahla.clear()

        except KeyError:
            #print(row['file'])
            #print(row['utterance_id'])SS
            list_shahla.clear()
            continue
    print("test")
    # print(df_agreement.iloc[count]['file'])
    # print(df_agreement.iloc[count]['utterance'])
    # print(df_agreement.iloc[track]['file'])
    # print(df_agreement.iloc[track]['utterance'])
    print(count)
    print(track)

    sum=0
    length=0

    for state, capital in dic_dist.items():
        # print(state, ":", capital/count)

        if state in list:

            print("sum")
            sum+=(capital['*PAR']+capital['*INV'])
        dic_dist[state]['total']=((capital['*PAR']+capital['*INV'])/count)*100
        length+=(capital['*PAR']+capital['*INV'])
    print(sum)
    print(sum/count)
    print(length)
    print(length/count)
    print(dic_dist)


def filter_prob(Y_pred_ovr):
    index_list = [0, 1, 2, 3, 4, 5, 6, 7]
    for item in Y_pred_ovr:

        indices = [i for i, x in enumerate(item) if x >= .5]
        if len(indices) == 0:
            indices_max = [i for i, x in enumerate(item) if x == max(item)]
            item[indices_max[0]] = .5
        elif len(indices) > 1:
            check = all(item in index_list for item in indices)
            if not check:
                set1 = set(index_list)
                set2 = set(indices)
                if set1.intersection(set2):
                    exclude = set2.difference(set1)
                    for i, x in enumerate(exclude):
                        item[x] = .1
                else:
                    indices_max = [x for i, x in enumerate(indices) if item[x] == max(item)]
                    for i, x in enumerate(indices_max):
                        if i != 0:
                            item[x] = .1

    return Y_pred_ovr


def agreement(df_agreement,name1,name2):
    taskdata = []
    list = ['', 'Answer: t3', 'Answer: t2', 'Answer: t1', 'Answer: t4', 'Answer: t5', 'Answer: t6', 'Answer: t7',
            'Answer: t8']
    mina_data = None
    Shahla_data = None
    count = 0


    for index, row in df_agreement.iterrows():
        track=0

        print(count)
        if (pd.isnull(row[name2])):
            continue
        if row[name1] == '['']' or row[name2] == '['']':
            continue
        list_shahla.clear()
        list_mina.clear()
        preprocess(row[name2], 'f')

        preprocess(row[name1], 's')
        for item in list_mina:
            if item in list:
                track=1

                break
        for item in list_shahla:

            if item in list:
                track = 1

                break
        if track:
            continue







        mina_data = list_mina[0]
        Shahla_data = list_shahla[0]
        print("take")
        print(mina_data)
        print(Shahla_data)
        # #if row['Document'] in ['1.txt','10.txt'] and len(row['Position'].split(':')[1])>1:
        # for item in list_mina:
        #     if item in list_shahla:
        #         mina_data = item
        #         Shahla_data = item
        #         sum += 1
        #         sum_similar = sum_similar + 1
        #         break

        # break

        # row_count += 1
        taskdata.append([0, str(count), mina_data])
        taskdata.append([1, str(count), Shahla_data])

        count+=1

        # if count > 0:
        #     continue
        # print(list_mina)
        # print(list_shahla)
        # count+=1
        # if count>40:
        # break
    # # print(taskdata)
    ratingtask = AnnotationTask(data=taskdata)
    print("kappa " + str(ratingtask.kappa()))
    # print(sum)

def calculate_agreement(df_agreement,name1,name2):
    taskdata=[]
    list=['','Answer: t3','Answer: t2','Answer: t1','Answer: t4','Answer: t5','Answer: t6','Answer: t7','Answer: t8']
    mina_data=None
    Shahla_data=None
    sum=0
    sum_similar = 0
    row_count = 0
    
    for index, row in df_agreement.iterrows():
        count=0
        
        #strip_shahla=row['shahla'][1:-1].strip(',')
        #print(strip_mina)
        # if(row['file']==34 and row['utterance_id']==11):
        #     continue
        if(pd.isnull(row[name2])):
            continue
        if row[name1]=='['']' or row[name2]=='['']':
            continue
        # if row['file'] ==12:
        #     continue
        # if (row['file'] == 16):
        #     break
            
        #mina=preprocess(row['mina'])
        # print("row")
        # print(type(row))
        # print("mina")
        #
        # print(row['file'])
        # print(row['utterance_id'])
        # print(row)
        preprocess(row[name2],'f')
       
        preprocess(row[name1],'s')
        # if index==333 or index==368:
            # print(list_mina)
        #print(list_shahla)
        #print(list_mina)
        
        if len(list_mina)>1:
            for item in list_mina:
                # print("item %s" %(str(item)))
                if not item or item not in list:
                    count=1
                   
                    # print (index)
                    # print("sum %s"%(list_mina))
                    break
                
        
        print("count %d"%count)

        
        # print("flav :"+mina)
        # print("sha" +shahla)
        # if(len(mina_data)==0 or len(Shahla_data)==0):
        #     continue
        mina_data=list_mina[0]
        Shahla_data=list_shahla[0]
        # #if row['Document'] in ['1.txt','10.txt'] and len(row['Position'].split(':')[1])>1: 
        for item in list_mina:
            if item in list_shahla:
                mina_data=item
                Shahla_data=item
                sum+=1
                sum_similar=sum_similar+1
                break
         
        #break        
          
        row_count+=1
        taskdata.append([0,str(index),mina_data])
        taskdata.append([1,str(index),Shahla_data]) 
        list_shahla.clear() 
        list_mina.clear()
        if count>0:
            continue
        #print(list_mina)
        #print(list_shahla)
        #count+=1
        # if count>40:
            # break
    # # print(taskdata)
    ratingtask = agreement.AnnotationTask(data=taskdata)
    print("kappa " +str(ratingtask.kappa()))
    print(sum)
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
def create_curation(df_agreement,name1,name2):
    taskdata = []
    list = ['', 'Answer: t3', 'Answer: t2', 'Answer: t1', 'Answer: t4', 'Answer: t5', 'Answer: t6', 'Answer: t7',
            'Answer: t8']
    mina_data = None
    Shahla_data = None
    sum = 0
    sum_similar = 0
    row_count = 0

    for index, row in df_agreement.iterrows():
        count = 0

        if (row['file'] == 'all' ):
            print("break")
            break
        if (pd.isnull(row['adjudicated label_new(shahla & mina)'])):

            if row[name1] == '['']' or row[name2] == '['']':
                taskdata.append('['']')

            preprocess(row[name2], 'f')

            preprocess(row[name1], 's')
            final_set=Union(list_shahla,list_mina)

            # print(index)
            # print(final_set)
            taskdata.append(final_set)
        else:
            # print(row['adjudicated label_new(shahla & mina)'])
            taskdata.append(row['adjudicated label_new(shahla & mina)'])

    columns={'curated':taskdata}
    cookie_Frame= pd.DataFrame(columns)
    return cookie_Frame

def parse_json(data,file_no):
    count=0
    start=None
    ends=None
    index=None
    
    for (k, v) in data.items():
        
        if k in ["_views"]: 
            for (key, value) in v.items():
                if key in ["_InitialView"]:
                    for (comp, v) in value.items():
                        if comp in ["Da_act"]:
                            
                            for val in v:
                                
                                if  val['end']==ends and count>0:
                                #print("Key: " + comp)
                                    ends = val['end']
                                    try:
                                        label[index].append(val['da_type'])
                                    except KeyError:
                                        # label[index].append("")
                                        print("error")
                                         
                                    
                                elif val['end']!=ends and count>=0:
                                    
                                    # file_id.append(file_no)
                                    # utterance_id.append(count)
                                    try:
                                    
                                        label.append([val['da_type']])
                                        file_id.append(file_no)
                                        utterance_id.append(count)
                                        index=len(label)-1
                                        ends = val['end']
                                        count+=1
                                    except KeyError:
                                        # label.append([""])
                                        print("error")
                                    # index=len(label)-1
                                    # ends = val['end']
                                    # count+=1
                                    
                            break
                    break
            break
                                    
                                    
                                    
                
                            
def get_json_data():
    for folder in ["mina_new"]:
        file_id.clear()
        utterance_id.clear()
        label.clear()

        PATH = "./data/" + folder
        print("hello")
        for path, dirs, files in os.walk(PATH):
            file_no=1
            for filename in files:


                print(filename)
                fullpath = os.path.join(path, filename)
                
                with open(fullpath, 'r',encoding="utf8")as input_file:

                    data = json.load(input_file)
                    parse_json(data,filename.split('.')[0])


                    file_no+=1
                    
        columns={'id':file_id,'utterance_id':utterance_id,'label':label}

        cookie_Frame= pd.DataFrame(columns)
        # cookie_Frame=cookie_Frame.append(pd.Series([id,label,repeat_sentence]))
        # with open('./data/json_utterance_mina.pickle', 'wb') as f:
        #     pickle.dump(cookie_Frame, f)
        # with open('./data/json_utterance_mina.pickle', 'rb') as f:
        #     cookie_Frame=pickle.load( f)
        print("data")
        print(cookie_Frame.head(5))
        writer = pd.ExcelWriter("./data/json_utterance_mina_new_new"+folder+".xlsx", engine='xlsxwriter')
        cookie_Frame.to_excel(writer, sheet_name='Sheet1',columns=['id','utterance_id','label'])
        writer.save()
def read_excel(df_excel):
    list1=['1.txt','2.txt','3.txt','4.txt','5.txt','6.txt','7.txt','8.txt','9.txt','10.txt','11.txt','12.txt','13.txt','14.txt','15.txt']
    list2=['16.txt','17.txt','18.txt','19.txt','20.txt','21.txt','22.txt','23.txt','24.txt','25.txt','26.txt','27.txt','28.txt','29.txt','30.txt','31.txt','32.txt','33.txt',
    '34.txt','35.txt']
    utterance=[]
    id=[]
    labels=[]
    file=[]
    ind=0
    for index, row in df_excel.iterrows():
        # if row['file'] in list1 or row['file'] in list2:
            # #file_no=row['utterance'].split('.')[0]
            # # if int(file_no)>35:
                # # break
            # if int(file_no)>35:
                # break
            # count=0
        #else:
            #print(row['utterance'])
        #if len(row['utterance'])>0 and len(row['utterance'].split(':')[1])>0:
        utterance.append(row['utterance'])
        #id.append(count)
        #file.append(file_no)
        file.append(row['file'])
        #count+=1
        labels.append([])
        if not pd.isnull(row['label1']):
            labels[ind].append(str(row['label1']))
            if not pd.isnull(row['label2']):
                labels[ind].append(str(row['label2']))
                if not pd.isnull(row['label3']):
                    labels[ind].append(str(row['label3']))
                    if not pd.isnull(row['label4']):
                        labels[ind].append(str(row['label4']))
                        if not pd.isnull(row['label5']):
                            labels[ind].append(str(row['label5']))
                            if not pd.isnull(row['label6']):
                                labels[ind].append(str(row['label6']))
                                if not pd.isnull(row['label7']):
                                    labels[ind].append(str(row['label7']))
                                        
                    
    
    
        ind+=1
    columns={'file':file,'label':labels,'utterance':utterance}  
    cookie_Frame= pd.DataFrame(columns)   
    # cookie_Frame=cookie_Frame.append(pd.Series([id,label,repeat_sentence]))            
    writer = pd.ExcelWriter("./result/full_agreement_shahla.xlsx", engine='xlsxwriter') 
    cookie_Frame.to_excel(writer, sheet_name='Sheet1',columns=['file','label','utterance'])        
    writer.save()      
def main():
    Y_pred=[[.5,.4,.2,.1,.9,.2,.1,.08,.5,.4,.2,.1,.9,.2,.1,.08],
            [.04,.4,.2,.1,.09,.2,.1,.08,.5,.4,.2,.9,.9,.2,.1,.8],
            [.5,.4,.2,.1,.9,.2,.1,.08,.05,.4,.2,.1,.09,.2,.1,.08],
            [.05,.4,.2,.1,.09,.2,.1,.08,.4,.05,.2,.1,.09,.2,.1,.08]]
    print(filter_prob(Y_pred))

    # get_json_data()
    # df_excel = pd.read_excel('./Dataset/annotation_shahla.xlsx', sheetname='Sheet1')
    # read_excel(df_excel)

    # df_agreement = pd.read_excel('./data/agreement_mina_shahla.xlsx', sheet_name='Sheet1')
    # # cookie_Frame=create_curation(df_agreement,"shahla","mina")
    # # writer = pd.ExcelWriter("./data/curated.xlsx", engine='xlsxwriter')
    # # cookie_Frame.to_excel(writer, sheet_name='Sheet1', columns=['curated'])
    # # writer.save()
    # # df_agreement = pd.read_excel('./data/full_agreement.xlsx', sheet_name='Sheet1')
    # # with open('./data/full_agreement_both.pickle', 'wb') as f:
    # #     pickle.dump(df_agreement, f)
    # # with open('./data/full_agreement_wide.pickle', 'rb') as f:
    # #     df_wide=pickle.load( f)
    # # print(df_wide.iloc[0:3,0:10])
    # df_wide=make_wide_dataframe(df_agreement)
    #
    # # print(len(df_wide))
    #
    # with open('./data/full_agreement_wide_extended.pickle', 'wb') as f:
    #     pickle.dump(df_wide, f)
    # with open('./data/full_agreement_wide_extended.pickle', 'rb') as f:
    #     df_wide=pickle.load(f)
    # print(len(df_wide))
    # print(df_wide.iloc[1421,0:25])
    # print(df_wide.iloc[1421]['utterance'])
    # print(df_wide.iloc[1421]['file'])
    # print(df_wide.iloc[1421]['utterance_id'])
    # print(df_wide.iloc[106, 0:25])
    # print(df_wide.iloc[106]['utterance'])
    # print(df_wide.columns.values.tolist())
    # print(df.file.unique())
    # print(len(df.file.unique()))
    # agreement(df_agreement,"shahla","mina")
    # calculate_agreement(df_agreement,"shahla","mina_new")
    # df_agreement = pd.read_excel('./Dataset/full_agreement_curated.xlsx', sheetname='Sheet1')
    # distribution(df_agreement)
if __name__=="__main__":
    main()
   
