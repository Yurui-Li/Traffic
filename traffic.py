# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import re
from copy import deepcopy

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
    #year.append(int(a[0]))
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
'''mean_time={}
road_id=data['link_ID'].unique()
for id in road_id:
    tem=data[data['link_ID']==id]
    aver_time=sum(list(tem['travel_time']))/len(tem)
    mean_time[id]=aver_time'''
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
    
#估算道路通行能力
in_link=list(data['in_links'])
out_link=list(data['out_links'])
length=list(data['length'])
width=list(data['width'])
capability=[]
for i in range(len(data)):
    #print(i)
    in_name=re.split(r'\#',str(in_link[i]))
    out_name=re.split(r'\#',str(out_link[i]))
    m_c_in=0
    for j in in_name:
        if str(j)!='nan':
            m_c_in+=mean_time[j]
    time_in=m_c_in/len(in_name)
    m_c_out=0
    for k in out_name:
        if str(k)!='nan':
            m_c_out+=mean_time[k]
    time_out=m_c_out/len(out_name)
    ca=0.2*width[i]+0.2*length[i]/10.0+0.15*time_in+0.15*time_out+0.3*aver_speed[i]
    capability.append(ca)
data['capability']=capability   

#丢弃无用特征
data=data.drop(['link_class','in_links','out_links',],axis=1)

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
#计算每个小时内拥堵级别1,2,3,4出现的lnodds值
hour_unique=sorted(data['s_hour'].unique())
hour_count=data.groupby(['s_hour']).size()
jam_class_count=data.groupby(['jam_class']).size()
H_J_count=data.groupby(['s_hour','jam_class']).size()
default_lnodds=np.log(jam_class_count/len(data))-np.log(1.0-jam_class_count/float(len(data)))
#构建存储lnodds值的字典
lnodds={}
lnoddsPA={}
MIN_CAT_COUNTS=4
for h in hour_unique:
    PA=hour_count[h]/float(len(data))
    lnoddsPA[h]=np.log(PA)-np.log(1.-PA)
    lnodds[h]=deepcopy(default_lnodds)
    for cl in H_J_count[h].keys():
        if (H_J_count[h][cl]>MIN_CAT_COUNTS) and (H_J_count[h][cl]<hour_count[h]):
            PA=H_J_count[h][cl]/float(hour_count[h])
            lnodds[h][cl]=np.log(PA)-np.log(1.0-PA)
    lnodds[h]=pd.Series(lnodds[h])
#构建lnodds特征    
hour_features=list(data['s_hour'])
ln=[]
for i in range(len(data)):
    ln.append(lnodds[hour_features[i]])
lnodds1=[];lnodds2=[];lnodds3=[];lnodds4=[]
for j in range(1,len(ln[0])+1):
    t=[]
    for i in range(len(ln)):
        t.append(ln[i][j])
    if j==1:
        lnodds1=t
    elif j==2:
        lnodds2=t
    elif j==3:
        lnodds3=t
    elif j==4:
        lnodds4=t
data['lnodds1']=lnodds1
data['lnodds2']=lnodds2
data['lnodds3']=lnodds3
data['lnodds4']=lnodds4
#构建lnoddsPA特征
lnPA=[]
for i in range(len(data)):
    lnPA.append(lnoddsPA[hour_features[i]])
data['lnoddsPA']=lnPA
#计算INRIX指标
inrix=[]
for i in range(len(data)):
    inrix.append(aver_speed[i]/immediate_velocity[i]-1)
data['inrix']=inrix
#data.to_csv('E:/train_data.csv',header=True,index=True,sep=',',line_terminator='\n')
'''
#分别计算工作日和周末的平均通过时间以及是否是上班必经路,暂时没用上
road_id=data['link_ID'].unique()
aver_time_weekday={}
aver_time_weekend={}
is_necessary={}
for id in road_id:
    tem=data[data['link_ID']==id]
    weekday_tem=tem[tem['Weekday']<6]
    weekend_tem=tem[tem['Weekday']>5]
    weekday_time=sum(list(weekday_tem['travel_time']))/len(weekday_tem)
    weekend_time=sum(list(weekend_tem['travel_time']))/len(weekend_tem)
    aver_time_weekday[id]=weekday_time
    aver_time_weekend[id]=weekend_time
    if abs(weekday_time-weekend_time)>5:
        is_necessary[id]=1
    else:
        is_necessary[id]=0
#构建是否拥堵的特征和是否是上班必经路的特征
is_jam=[]
is_nece=[]
for i in range(len(data)):
    id_tem=ID[i]
    is_nece.append(is_necessary[id_tem])
    if Weekday[i]<6:
        if travel_time[i]>aver_time_weekday[id_tem]:
            is_jam.append(1)
        else:
            is_jam.append(0)
    else:
        if travel_time[i]>aver_time_weekend[id_tem]:
            is_jam.append(1)
        else:
            is_jam.append(0)
data['is_jam']=is_jam
data['is_nece']=is_nece
'''#暂时没用上





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
    
#估算道路通行能力
in_link=list(test_data['in_links'])
out_link=list(test_data['out_links'])
length=list(test_data['length'])
width=list(test_data['width'])
capability_test=[]
for i in range(len(test_data)):
    in_name=re.split(r'\#',str(in_link[i]))
    out_name=re.split(r'\#',str(out_link[i]))
    m_c_in=0
    for j in in_name:
        if str(j)!='nan':
            m_c_in+=mean_time[j]
    time_in=m_c_in/len(in_name)
    m_c_out=0
    for k in out_name:
        if str(k)!='nan':
            m_c_out+=mean_time[k]
    time_out=m_c_out/len(out_name)
    ca=0.2*width[i]+0.2*length[i]/10.0+0.15*time_in+0.15*time_out+0.3*aver_speed_t[i]
    capability_test.append(ca)
test_data['capability']=capability_test             

#丢弃无用特征
test_data=test_data.drop(['in_links','out_links',],axis=1)

#用平均通过时间代替实际通过时间来进行特征构建
s_hour_t=list(test_data['s_hour'])
s_minute_t=list(test_data['s_minute'])
e_hour_t=list(test_data['e_hour'])
e_minute_t=list(test_data['e_minute'])
weekday_t=list(test_data['Weekday'])
sum_t=[]
for i in range(len(test_data)):
    temp=data[data['link_ID']==id_link[i]]
    temp=temp[temp['s_hour']==s_hour_t[i]]
    temp=temp[temp['s_minute']==s_minute_t[i]]
    temp=temp[temp['e_hour']==e_hour_t[i]]
    temp=temp[temp['e_minute']==e_minute_t[i]]
    temp=temp[temp['Weekday']==weekday_t[i]]
    if len(temp)!=0:
        s=sum(list(temp['travel_time']))/len(temp)
        sum_t.append(s)
    elif len(temp)==0:
        sum_t.append(0)
    print(i)

m=sum(sum_t)/len(sum_t)
for i in range(len(test_data)):
    if sum_t[i]==0.0:
        sum_t[i]=m
test_data['travel_time']=sum_t
'''s_hour=list(data['s_hour'])
s_minute=list(data['s_minute'])
e_hour=list(data['e_hour'])
e_minute=list(data['e_minute'])
s_hour_t=list(test_data['s_hour'])
s_minute_t=list(test_data['s_minute'])
e_hour_t=list(test_data['e_hour'])
e_minute_t=list(test_data['e_minute'])
weekday_t=list(test_data['Weekday'])
sum_t=[]
ss=0.0
c=0
for i in range(len(test_data)):
    tt=[]
    for j in range(len(data)):
        if (ID[j]==id_link[i])&(s_hour[j]==s_hour_t[i])&(s_minute[j]==s_minute_t[i])&(e_hour[j]==e_hour_t[i])&(e_minute[j]==e_minute_t[i])&(Weekday[j]==weekday_t[i]):
            #tt.append(travel_time[j])
            ss+=travel_time[j]
            c+=1
    sum_t.append(ss/c)
    print(i)
test_data['travel_time']=sum_t'''
'''#计算道路拥堵级别和即时速度
jam_class_t=[]
immediate_velocity_t=[]
for i in range(len(test_data)):
    speed_phrase=length[i]/sum_t[i]*3.6
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
#计算每个小时内拥堵级别1,2,3,4出现的lnodds值
hour_unique_t=sorted(test_data['s_hour'].unique())
hour_count_t=test_data.groupby(['s_hour']).size()
jam_class_count_t=test_data.groupby(['jam_class']).size()
H_J_count_t=test_data.groupby(['s_hour','jam_class']).size()
default_lnodds_t=np.log(jam_class_count_t/len(test_data))-np.log(1.0-jam_class_count_t/float(len(test_data)))
#构建存储lnodds值的字典
lnodds_t={}
lnoddsPA_t={}
#MIN_CAT_COUNTS=4
for h in hour_unique_t:
    PA=hour_count_t[h]/float(len(test_data))
    lnoddsPA_t[h]=np.log(PA)-np.log(1.0-PA)
    lnodds_t[h]=deepcopy(default_lnodds_t)
    for cl in H_J_count_t[h].keys():
        if (H_J_count_t[h][cl]>MIN_CAT_COUNTS) and (H_J_count_t[h][cl]<hour_count_t[h]):
            PA=H_J_count_t[h][cl]/float(hour_count_t[h])
            lnodds_t[h][cl]=np.log(PA)-np.log(1.0-PA)
    lnodds_t[h]=pd.Series(lnodds_t[h])
#构建lnodds特征    
hour_features_t=list(test_data['s_hour'])
ln_t=[]
for i in range(len(test_data)):
    ln_t.append(lnodds_t[hour_features_t[i]])
for i in range(len(ln_t)):
    ln_t[i][1]=0.0
lnodds1_t=[];lnodds2_t=[];lnodds3_t=[];lnodds4_t=[]
for j in range(1,len(ln_t[0])+1):
    t=[]
    for i in range(len(ln_t)):
        t.append(ln_t[i][j])
    if j==1:
        lnodds1_t=t
    elif j==2:
        lnodds2_t=t
    elif j==3:
        lnodds3_t=t
    elif j==4:
        lnodds4_t=t
test_data['lnodds1']=lnodds1_t
test_data['lnodds2']=lnodds2_t
test_data['lnodds3']=lnodds3_t
test_data['lnodds4']=lnodds4_t
lnodds1_t=[]
for i in range(len(test_data)):
    lnodds1_t.append((lnodds2_t[i]+lnodds3_t[i]+lnodds4_t[i])/3.0)
test_data['lnodds1']=lnodds1_t
#构建lnoddsPA特征
lnPA_t=[]
for i in range(len(test_data)):
    lnPA_t.append(lnoddsPA_t[hour_features_t[i]])
test_data['lnoddsPA']=lnPA_t
#计算INRIX指标
aver_speed_t=list(test_data['aver_speed'])
inrix_t=[]
for i in range(len(test_data)):
    inrix_t.append(aver_speed_t[i]/immediate_velocity_t[i]-1.0)
test_data['inrix']=inrix_t

test_data=test_data.drop('travel_time',axis=1)
test_data.to_csv('E:/test_data_final.csv',header=True,index=True,sep=',',line_terminator='\n')'''
'''
#构建是否拥堵的特征和是否是上班必经路的特征，暂时没用上
is_jam_t=[]
is_nece_t=[]
ID_t=list(test_data['link_ID'])
Weekday_t=list(test_data['Weekday'])
travel_time_t=list(test)
for i in range(len(test_data)):
    id_tem=ID_t[i]
    is_nece_t.append(is_necessary[id_tem])
    if Weekday_t[i]<6:
        if travel_time[i]>aver_time_weekday[id_tem]:
            is_jam_t.append(1)
        else:
            is_jam_t.append(0)
    else:
        if travel_time[i]>aver_time_weekend[id_tem]:
            is_jam_t.append(1)
        else:
            is_jam_t.append(0)
test_data['is_jam']=is_jam_t
test_data['is_nece']=is_nece_t
'''

'''
#拆分训练集,做本地错误预估
from sklearn.cross_validation import train_test_split
training, validation = train_test_split(data, train_size=0.8)
training_label=np.array(training['travel_time'])
validation_label=np.array(validation['travel_time'])
training=training.drop('travel_time',axis=1) 
validation=validation.drop('travel_time',axis=1)
training['t']=training_label
validation['t'] =validation_label'''

'''from sklearn import preprocessing   
label = preprocessing.LabelEncoder()
label.fit(training.link_ID)
link_id_train=label.transform(training.link_ID)
training['link_ID']=link_id_train
link_id_test=label.transform(validation.link_ID)
validation['link_ID']=link_id_test'''

'''
from sklearn import preprocessing
feature_train=training.columns.tolist()
feature_validation=validation.columns.tolist()
scaler=preprocessing.StandardScaler()
for i in feature_train[1:len(feature_train)-1]:
    scaler.fit(training[i])
    training[i]=scaler.transform(training[i])
for i in feature_validation[1:len(feature_validation)-1]:
    scaler.fit(validation[i])
    validation[i]=scaler.transform(validation[i])'''

'''
#ElasticNet模型
from sklearn import linear_model
model=linear_model.ElasticNet(alpha=0.3,l1_ratio=0.5)
model.fit(training, training_label)
ttp=list(model.predict(validation))
ttr=list(validation_label)
sum=0.0
for i in range(len(ttp)):
    sum+=abs(ttp[i]-ttr[i])/ttr[i]
mape=sum/len(ttr)
print(mape)
'''
'''
#为每条路训练一个模型
from sklearn import linear_model
road_id=training['link_ID'].unique()
for id in road_id:
    print(id)
    training_tem=training[training['link_ID']==id].copy()
    validation_tem=validation[validation['link_ID']==id].copy()
    training_label_tem=training_tem['t']
    validation_label_tem=validation_tem['t']
    training_tem=training_tem.drop(['link_ID','t'],axis=1)
    validation_tem=validation_tem.drop(['link_ID','t'],axis=1)
    print(training_tem.shape,validation_tem.shape)
    model_tem=linear_model.ElasticNet(alpha=0.3)
    model_tem.fit(training_tem,training_label_tem)
    ttp=list(model_tem.predict(validation_tem))
    ttr=list(validation_label_tem)
    sum=0.0
    for i in range(len(ttp)):
        sum+=abs(ttp[i]-ttr[i])/ttr[i]
    mape=sum/len(ttr)
    print(mape)'''

'''
#神经网络模型
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(50,input_dim=16,init='uniform',activation='relu'))
model.add(Dense(25,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform'))
model.compile(loss='mape',optimizer='sgd',metrics=['accuracy'])
model.fit(training, training_label,nb_epoch=20,batch_size=640)
ttp=list(model.predict(validation))
ttr=list(validation_label)
sum=0.0
for i in range(len(ttp)):
    sum+=abs(ttp[i]-ttr[i])/ttr[i]
mape=sum/len(ttr)
print(mape)
'''


test_data=pd.read_csv('E:/test_data_final.txt',sep=',')
test_data=test_data.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1) 
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

        
#对link_ID特征进行编码   
from sklearn import preprocessing   
#label = preprocessing.LabelEncoder()
#label.fit(data.link_ID)
#link_id_train=label.transform(data.link_ID)
#data['link_ID']=link_id_train
#link_id_test=label.transform(test_data.link_ID)
#test_data['link_ID']=link_id_test

#对特征进行正则化
feature_list=data.columns.tolist()
scaler=preprocessing.StandardScaler()
for i in feature_list[1:len(feature_list)-1]:
    scaler.fit(data[i])
    data[i]=scaler.transform(data[i])
for i in feature_list[1:len(feature_list)-1]:
    scaler.fit(test_data[i])
    test_data[i]=scaler.transform(test_data[i])


    


#构建全局训练模型
links_id=list(merge_data['link_ID'])
test_label=[]
from sklearn import linear_model
for id in links_id:
    data_tem=data[data['link_ID']==id]
    test_data_tem=test_data[test_data['link_ID']==id]
    data_label_tem=data_tem['t']
    data_tem=data_tem.drop('t',axis=1)
    print(id,len(data_tem),len(test_data_tem))
    '''model=linear_model.ElasticNet(alpha=0.3)
    model.fit(data_tem,data_label_tem)
    result=list(model.predict(test_data_tem))
    for i in range(len(result)):
        test_label.append(result[i])'''




#输出结果
upload['travel_time']=test_label
upload.to_csv('E:/result.csv',header=False,index=False,sep='#',line_terminator='\n')






































