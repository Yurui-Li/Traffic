# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 19:43:23 2017

@author: ynu
"""

import pandas as pd
import numpy as np
import re
#from copy import deepcopy

#计算周几的函数，基于基姆拉尔森计算公式
def CalculateWeekDay(year,month,day):
    if (month==1)|(month==2):
        month+=12
        year-=1  
    iWeek = int((day+2*month+3*(month+1)/5+year+year/4-year/100+year/400)%7)
    return iWeek+1

#读取文件
link_info=pd.read_csv('E:\gy_contest_link_info.txt',sep=';')
link_top=pd.read_csv('E:\gy_contest_link_top.txt',sep=';')
train_data=pd.read_csv('E:\gy_contest_link_traveltime_training_data.txt',sep=';')
#拆分列time_interval
time_interval=list(train_data['time_interval'])
interval_s=[]
interval_e=[]
for i in range(len(train_data)):
    a=re.split(r'[\s\,]',time_interval[i])
    a[3]=a[3][:-1]
    interval_s.append(a[1])
    interval_e.append(a[3])
                 
train_data['interval_s']=interval_s
train_data['interval_e']=interval_e
#拆分列interval_s、interval_e
start_time=list(train_data['interval_s'])
end_time=list(train_data['interval_e'])
s_hour=[];e_hour=[]
s_minute=[];e_minute=[]
for i in range(len(train_data)):
    s_tem=re.split(r'\:',start_time[i])
    e_tem=re.split(r'\:',end_time[i])
    s_hour.append(float(s_tem[0]))
    s_minute.append(float(s_tem[1]))
    e_hour.append(float(e_tem[0]))
    e_minute.append(float(e_tem[1]))
    
train_data['s_hour']=s_hour
train_data['s_minute']=s_minute
train_data['e_hour']=e_hour
train_data['e_minute']=e_minute
#丢弃冗余列        
train_data=train_data.drop(['time_interval','interval_s','interval_e'],axis=1)
#处理link_top
in_link=list(link_top['in_links'])
out_link=list(link_top['out_links'])
in_num=[]
for i in range(len(link_top)):
    tem=re.split(r'\#',str(in_link[i]))
    if str(tem[0])=='nan':
        in_num.append(0)
    else:
        in_num.append(len(tem))     
out_num=[]
for i in range(len(link_top)):
    tem=re.split(r'\#',str(out_link[i]))
    if str(tem[0])=='nan':
        out_num.append(0)
    else:
        out_num.append(len(tem))
link_top['in_num']=in_num
link_top['out_num']=out_num
#合并link_info和link_top
merge_data=pd.merge(link_info,link_top,on='link_ID')
#合并所有训练数据
data=pd.merge(train_data,merge_data,on='link_ID')
#拆分datelie
date=list(data['date'])
month=[]
day=[]
for i in range(len(data)):
    a=re.split(r'\-',date[i])
    month.append(int(a[1]))
    day.append(int(a[2]))
data['month']=month
data['day']=day

#计算星期几
Weekday=[]
for i in range(len(date)):
    t=re.split(r'\-',date[i])
    year_c=int(t[0]);month_c=int(t[1]);day_c=int(t[2])
    Weekday.append(CalculateWeekDay(year_c,month_c,day_c))
data['Weekday']=Weekday
data=data.drop(['date'],axis=1)
#判断是否是上班高峰期
hours=list(data['s_hour'])
Weekday=list(data['Weekday'])
isbusy=[]
for i in range(len(hours)):
    tem_h=int(hours[i])
    tem_w=int(Weekday[i])
    if (tem_w<6)&(((tem_h>=7)&(tem_h<=9))|((tem_h>=17)&(tem_h<=19))):
        isbusy.append(1)
        
    else:
        isbusy.append(0)
data['isbusy']=isbusy
#计算每条路的平均通过时间
ID=list(data['link_ID'])
ID_unique=list(link_info['link_ID'])
travel_time=list(data['travel_time'])
mean_time={}
for id in ID_unique:
    count=0
    sum_time=0
    for j in range(len(ID)):
        if id==ID[j]:
            sum_time+=travel_time[j]
            count+=1
    mean=sum_time/float(count)
    mean_time[id]=mean
#计算每条记录的平均速度
length=list(data['length'])
id_link=list(data['link_ID'])
aver_speed=[]
overspeed=[]
for i in range(len(data)):
    speed_instance=length[i]/mean_time[id_link[i]]*3.6
    aver_speed.append(speed_instance)
    if speed_instance>=40.0:
        overspeed.append(1)
    else:
        overspeed.append(0)
data['aver_speed']=aver_speed
data['overspeed']=overspeed


#计算道路拥堵级别和即时速度
jam_class=[]
immediate_velocity=[]
for i in range(len(data)):
    speed_phrase=length[i]/travel_time[i]*3.6
    immediate_velocity.append(speed_phrase)
    if int(speed_phrase)<10:
        jam_class.append(4)
    elif (int(speed_phrase)>=10)&(int(speed_phrase)<20):
        jam_class.append(3)
    elif (int(speed_phrase)>=20)&(int(speed_phrase)<30):
        jam_class.append(2)
    elif int(speed_phrase)>=30:
        jam_class.append(1)
    else:
        jam_class.append(0)
data['jam_class']=jam_class
data['immediate_velocity']=immediate_velocity
#计算INRIX指标
inrix=[]
for i in range(len(data)):
    inrix.append(aver_speed[i]/immediate_velocity[i]-1)
data['inrix']=inrix

#估算道路通行能力
in_link=list(data['in_links'])
out_link=list(data['out_links'])
length=list(data['length'])
width=list(data['width'])
time_in=[]
time_out=[]
capability=[]
for i in range(len(data)):
    #print(i)
    in_name=re.split(r'\#',str(in_link[i]))
    out_name=re.split(r'\#',str(out_link[i]))
    m_c_in=0
    for j in in_name:
        if str(j)!='nan':
            m_c_in+=mean_time[j]
    time_in_per=m_c_in/len(in_name)
    time_in.append(time_in_per)
    m_c_out=0
    for k in out_name:
        if str(k)!='nan':
            m_c_out+=mean_time[k]
    time_out_per=m_c_out/len(out_name)
    time_out.append(time_out_per)
    ca=0.2*width[i]+0.2*length[i]/10.0+0.1*time_in_per+0.1*time_out_per+0.1*aver_speed[i]+0.1*jam_class[i]+0.1*immediate_velocity[i]+0.1*inrix[i]
    capability.append(ca)
data['time_in']=time_in
data['time_out']=time_out
data['capability']=capability   

#判断拥堵，即时速度小于0.8倍平均速度的视为拥堵
jam=[]
for i in range(len(data)):
    if immediate_velocity[i]<0.8*aver_speed[i]:
        jam.append(1)
    else:
        jam.append(0)
data['jam']=jam

#丢弃无用特征
data=data.drop(['link_class','in_links','out_links',],axis=1)



#创建测试集
test_data=pd.DataFrame()
s_hour=[]
s_minute=[]
e_hour=[]
e_minute=[]
day=[]   
for i in range(1,31):
    for k in range(0,60,2):
        day.append(i)
        if k==58:
            e_hour.append(9.0)
            e_minute.append(0.0)
            s_hour.append(8.0)
            s_minute.append(float(k))
        else:
            e_hour.append(8.0)
            e_minute.append(float(k+2))
            s_hour.append(8.0)
            s_minute.append(float(k))                
links_id=list(merge_data['link_ID'])
length=list(merge_data['length'])
width=list(merge_data['width'])
in_num=list(merge_data['in_num'])
out_num=list(merge_data['out_num'])
in_links=list(merge_data['in_links'])
out_links=list(merge_data['out_links'])
for i in range(len(merge_data)):
    weekday=[]
    tem=pd.DataFrame(index=np.arange(900))
    tem['link_ID']=links_id[i]
    tem['s_hour']=s_hour
    tem['s_minute']=s_minute
    tem['e_hour']=e_hour
    tem['e_minute']=e_minute
    tem['length']=length[i]
    tem['width']=width[i]
    tem['in_num']=in_num[i]
    tem['out_num']=out_num[i]
    tem['month']=6
    tem['day']=day
    for j in range(900):
        weekday.append(CalculateWeekDay(2016,6,day[j]))
    tem['Weekday']=weekday
    isbusy=[]
    for k in range(900):
        tem_h=int(s_hour[k])
        tem_w=int(weekday[k])
        if (tem_w<6)&(((tem_h>=7)&(tem_h<=9))|((tem_h>=17)&(tem_h<=19))):
            isbusy.append(1)
        else:
            isbusy.append(0)
    tem['isbusy']=isbusy
    tem['in_links']=in_links[i]
    tem['out_links']=out_links[i]
    test_data=test_data.append(tem,ignore_index=True)

#计算每条记录的平均速度
length=list(test_data['length'])
id_link=list(test_data['link_ID'])
aver_speed_t=[]
overspeed_t=[]
for i in range(len(test_data)):
    speed_instance=length[i]/mean_time[id_link[i]]*3.6
    aver_speed_t.append(speed_instance)
    if speed_instance>=40.0:
        overspeed_t.append(1)
    else:
        overspeed_t.append(0)
test_data['aver_speed']=aver_speed_t
test_data['overspeed']=overspeed_t
    
             


#使用之前成绩最好的预测结果来构建新特征
train=pd.read_csv('E:/li3.txt',header=None,sep=',')
travel_time_t=list(train.ix[:,0])
#计算道路拥堵级别和即时速度
jam_class_t=[]
immediate_velocity_t=[]
for i in range(len(test_data)):
    speed_phrase=length[i]/travel_time_t[i]*3.6
    immediate_velocity_t.append(speed_phrase)
    if int(speed_phrase)<10:
        jam_class_t.append(4)
    elif (int(speed_phrase)>=10)&(int(speed_phrase)<20):
        jam_class_t.append(3)
    elif (int(speed_phrase)>=20)&(int(speed_phrase)<30):
        jam_class_t.append(2)
    elif int(speed_phrase)>=30:
        jam_class_t.append(1)
    else:
        jam_class_t.append(0)
test_data['jam_class']=jam_class_t
test_data['immediate_velocity']=immediate_velocity_t
#计算INRIX指标
aver_speed_t=list(test_data['aver_speed'])
inrix_t=[]
for i in range(len(test_data)):
    inrix_t.append(aver_speed_t[i]/immediate_velocity_t[i]-1.0)
test_data['inrix']=inrix_t


#估算道路通行能力
in_link=list(test_data['in_links'])
out_link=list(test_data['out_links'])
length=list(test_data['length'])
width=list(test_data['width'])
time_in_t=[]
time_out_t=[]
capability_test=[]
for i in range(len(test_data)):
    in_name=re.split(r'\#',str(in_link[i]))
    out_name=re.split(r'\#',str(out_link[i]))
    m_c_in=0
    for j in in_name:
        if str(j)!='nan':
            m_c_in+=mean_time[j]
    time_in_per_t=m_c_in/len(in_name)
    time_in_t.append(time_in_per_t)
    m_c_out=0
    for k in out_name:
        if str(k)!='nan':
            m_c_out+=mean_time[k]
    time_out_per_t=m_c_out/len(out_name)
    time_out_t.append(time_in_per_t)
    ca=0.2*width[i]+0.2*length[i]/10.0+0.1*time_in_per_t+0.1*time_out_per_t+0.1*aver_speed_t[i]+0.1*jam_class_t[i]+0.1*immediate_velocity_t[i]+0.1*inrix_t[i]
    capability_test.append(ca)
test_data['time_in']=time_in_t
test_data['time_out']=time_out_t
test_data['capability']=capability_test

#判断拥堵，即时速度小于0.8倍平均速度的视为拥堵
jam_t=[]
for i in range(len(test_data)):
    if immediate_velocity_t[i]<0.8*aver_speed_t[i]:
        jam_t.append(1)
    else:
        jam_t.append(0)
test_data['jam']=jam_t

#丢弃无用特征
test_data=test_data.drop(['in_links','out_links',],axis=1)

#保存y
id_link=list(test_data['link_ID'])
train_label=data['travel_time']
data=data.drop('travel_time',axis=1)
data['t']=train_label  

#构建提交数据格式
day_test=list(test_data['day'])
column_2=[]
for i in range(len(test_data)):
    if day_test[i]<10:
        column_2.append('2016-06-0'+str(int(day_test[i])))
    else:
        column_2.append('2016-06-'+str(int(day_test[i])))
s_hour_tj=list(test_data['s_hour'])
s_minute_tj=list(test_data['s_minute'])
e_hour_tj=list(test_data['e_hour'])
e_minute_tj=list(test_data['e_minute'])
column_3=[]
for i in range(len(test_data)):
    if s_minute_tj[i]<10: 
        head='['+column_2[i]+' 0'+str(int(s_hour_tj[i]))+':0'+str(int(s_minute_tj[i]))+':00'
    else:
        head='['+column_2[i]+' 0'+str(int(s_hour_tj[i]))+':'+str(int(s_minute_tj[i]))+':00'
    if e_minute_tj[i]<10: 
        tail=','+column_2[i]+' 0'+str(int(e_hour_tj[i]))+':0'+str(int(e_minute_tj[i]))+':00)'
    else:
        tail=','+column_2[i]+' 0'+str(int(e_hour_tj[i]))+':'+str(int(e_minute_tj[i]))+':00)'
    column_3.append(head+tail)
upload=pd.DataFrame()
upload['link_ID']=id_link
upload['date']=column_2
upload['time_interval']=column_3


#对特征进行正则化
from sklearn import preprocessing   
feature_list=data.columns.tolist()
scaler=preprocessing.StandardScaler()
for i in feature_list[0:len(feature_list)-2]:
    scaler.fit(data[i])
    data[i]=scaler.transform(data[i])
for i in feature_list[0:len(feature_list)-2]:
    scaler.fit(test_data[i])
    test_data[i]=scaler.transform(test_data[i])
    

#分别为拥堵和不拥堵的建立模型    
test_label=[]
from sklearn import linear_model
for i in [0,1]:
    data_tem=data[data['jam']==i]
    test_data_tem=test_data[test_data['jam']==i]
    data_label_tem=data_tem['t']
    data_tem=data_tem.drop('t',axis=1)
    print(i,len(data_tem),len(test_data_tem))
    model=linear_model.ElasticNet(alpha=0.3,l1_ratio=0.5)
    model.fit(data_tem,data_label_tem)
    result=list(model.predict(test_data_tem))
    for i in range(len(result)):
        test_label.append(result[i])
'''
#构建全局训练模型,每个link_ID建立一个模型
links_id=list(merge_data['link_ID'])
test_label=[]
from sklearn import linear_model
for id in links_id:
    data_tem=data[data['link_ID']==id]
    test_data_tem=test_data[test_data['link_ID']==id]
    data_label_tem=data_tem['t']
    data_tem=data_tem.drop('t',axis=1)
    print(id,len(data_tem),len(test_data_tem))
    model=linear_model.ElasticNet(alpha=0.1,l1_ratio=0.9)
    model.fit(data_tem,data_label_tem)
    result=list(model.predict(test_data_tem))
    for i in range(len(result)):
        test_label.append(result[i])
'''
#输出结果
test_label=[abs(i) for i in test_label]
upload['travel_time']=test_label
upload.to_csv('E:/result.csv',header=False,index=False,sep='#',line_terminator='\n')

'''
#拆分训练集,做本地错误预估
from sklearn.cross_validation import train_test_split
training, validation = train_test_split(data, train_size=0.8)
training_label=np.array(training['travel_time'])
validation_label=np.array(validation['travel_time'])
training=training.drop('travel_time',axis=1) 
validation=validation.drop('travel_time',axis=1)
#training['t']=training_label
#validation['t'] =validation_label

from sklearn import preprocessing   
label = preprocessing.LabelEncoder()
label.fit(training.link_ID)
link_id_train=label.transform(training.link_ID)
training['link_ID']=link_id_train
link_id_test=label.transform(validation.link_ID)
validation['link_ID']=link_id_test

from sklearn import preprocessing   
feature_list=training.columns.tolist()
scaler=preprocessing.StandardScaler()
for i in feature_list:
    scaler.fit(training[i])
    training[i]=scaler.transform(training[i])
for i in feature_list:
    scaler.fit(validation[i])
    validation[i]=scaler.transform(validation[i])

from sklearn import linear_model
alpha=np.arange(0.1,1.6,0.1)
#mape=[]
score=[]
for i in alpha:
    model=linear_model.Ridge(alpha=i)
    model.fit(training,training_label)
    score.append(model.score(validation,validation_label))
    #ttp=list(model.predict(validation))
    #ttr=list(validation_label)
    #sum=0.0
    #for i in range(len(ttp)):
        #sum+=abs(ttp[i]-ttr[i])/ttr[i]
    #mape.append(sum/len(ttr))

import matplotlib.pyplot as plt
plt.plot(alpha,score)
'''
'''
#用平均通过时间代替实际通过时间来进行特征构建
id_link=list(test_data['link_ID'])
months=list(test_data['month'])
days=list(test_data['day'])
busy=list(test_data['isbusy'])
sum_t=[]
np.random.seed(1)
for i in range(len(test_data)):
    temp=data[data['link_ID']==id_link[i]]
    temp=temp[temp['month']==months[i]]
    temp=temp[temp['isbusy']==busy[i]]
    temp=temp[temp['day']==days[i]]   
    s=np.array(temp['travel_time'])
    mean_s=s.mean()
    std_s=s.std()
    if (len(s)!=0)&(std_s!=0.0):
        sum_t.append(mean_s+np.random.normal(loc=0.0,scale=std_s))
    else:
        sum_t.append(0)
    print(i)
#放宽条件处理0值
another=[]
for i in range(len(sum_t)):
    if sum_t[i]==0.0:
        another.append(i)
for j in another:
    hehe=data[data['link_ID']==id_link[j]]
    hehe=hehe[hehe['month']==months[j]]
    hehe=hehe[hehe['isbusy']==busy[j]]
    s_hehe=np.array(hehe['travel_time'])
    m_s=s_hehe.mean()
    s_s=s_hehe.std()
    if (len(s_hehe)!=0)&(s_s!=0.0):
        sum_t[j]=m_s+np.random.normal(loc=0.0,scale=s_s)
    else:
        sum_t[j]=0
        print(j)
    
#保存结果，因上述代码需要约8小时  
li=pd.DataFrame()
li['t']=sum_t
li.to_csv('E:/li3.csv',header=False,index=False,sep=',',line_terminator='\n')
'''
