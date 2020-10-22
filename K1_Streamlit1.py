# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:14:47 2020

@author: HornLa
"""

import streamlit as st
import quantstats as qs
import pandas as pd
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
#import quantstats as qs
import quandl
import yfinance as yf
import seaborn as sns




#import nsepy
#from nsepy import get_history
#from datetime import date
#import datetime
#from nsetools import Nse

st.write("""# KVANTIN 1 - NORDIC HEDGE""")
st.write("""### Developed by - Lars Horn @Kvantin""")


st.sidebar.title("Innskudd ved start")
# Add a slider to the sidebar:
start_cash=st.sidebar.slider(
    'Select a range of values',
    10000, 1000000,step=(10000))

period1 = st.sidebar.number_input('Momentum periode 1',5,100,(20))
st.write('Momentum periode 1 ', period1)

period2 = st.sidebar.number_input('Momentum periode 2',20,252,(30))
st.write('Momentum periode 2 ', period2)

start = st.sidebar.date_input("Select start date",datetime.date(2007, 3, 6))
end = st.sidebar.date_input("Select end date",datetime.datetime.now().date())

Momstart = st.sidebar.date_input('Momentum start date',datetime.date(2020, 5, 6))
#Momend = st.sidebar.date_input("Select end date",datetime.datetime.now().date())



OBX = yf.download('OBXEDNB.OL', start=start, end=None, parse_dates=True)
OBX.drop(['High', 'Low','Open','Volume','Close'], axis=1, inplace=True)
OBX.columns = OBX.columns.str.replace('Adj Close', 'OBX Close')
OBX = OBX.bfill()
OBX.index = pd.to_datetime(OBX.index)
#Copenhagen
OMXC=quandl.get("NASDAQOMX/OMXC20", authtoken="Jh2h4hQAEiQF7CVyHx7x",start_date=start)
OMXC = OMXC.rename({'Index Value': 'OMXC Close'}, axis=1)  # new method
OMXC.drop(['High', 'Low','Total Market Value','Dividend Market Value'], axis=1, inplace=True)
#Helesinki
OMXH=quandl.get("NASDAQOMX/OMXH25", authtoken="Jh2h4hQAEiQF7CVyHx7x", start_date=start)
OMXH = OMXH.rename({'Index Value': 'OMXH Close'}, axis=1)  # new method
OMXH.drop(['High', 'Low','Total Market Value','Dividend Market Value'], axis=1, inplace=True)
#Stockholm
OMXS=quandl.get("NASDAQOMX/OMXS30", authtoken="Jh2h4hQAEiQF7CVyHx7x" ,start_date=start)
OMXS = OMXS.rename({'Index Value': 'OMXS Close'}, axis=1)  # new method
OMXS.drop(['High', 'Low','Total Market Value','Dividend Market Value'], axis=1, inplace=True)
    
Ind = pd.concat([OMXC,OMXH, OBX, OMXS,], sort=True, axis=1)
Ind.bfill(inplace=True)
Ind.ffill(inplace=True)
    

@st.cache
def load_data():
    return Ind

data = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    st.line_chart(data)


#dataframes for momentum verdier
s1 = data / data.shift(period1)-1
s3 = data / data.shift(period2)-1

Sjekk1 = s1+s3
Sjekk1.fillna(0, inplace=True)
Sjekkny=Sjekk1[Momstart:]
st.line_chart(Sjekkny)

#Finner beste instrument og #lager en variabel som heter best
best = Sjekk1.idxmax(axis = 1, skipna=True)
#lager en dataframe som heter Best
Best = pd.DataFrame(index=Ind.index)
Best['Index'] = best.dropna()
Best['Max'] =  Sjekk1.max(axis=1)
Best['Index'] = np.where(Best['Max'] >= 0, Best['Index'], 'TLT Close')

Sjekk5= Sjekk1.tail()


if st.checkbox('Vis siste dager'):
    st.subheader('Siste dagers momentum')
    st.dataframe(Sjekk5.style.background_gradient(cmap='Greens', low=.4, high=0))

    
#Setter tall for beste indexs
conditions  = [(Best['Index'] == 'OMXC Close'),
               (Best['Index'] == 'OMXH Close'),
               (Best['Index'] == 'OBX Close'),
               (Best['Index'] == 'OMXS Close'),
               (Best['Index'] == 'TLT Close')]
    
choices     = [1.0, 2.0, 3.0, 4.0, 5.0]
    
Best['Tall'] = np.select(conditions, choices, default=0)
#Best['Tall'] = Best['Tall'].shift(1)
#Bruker bare denne når vi skal inn og ut 1 gang i måneden
Best['Month']   = pd.DatetimeIndex(Best.index).month
Best['Month_c'] = np.where(Best['Month'].diff() !=0 , 1 ,0 )
Best['Month_s'] = Best['Tall'] * Best['Month_c'].dropna()
    
#Bruker bare denne når vi skal inn og ut hver måned
test_ny = [Best.Month_s.values[0]]
    
var = None
for i in range(1, len(Best.index)):
    if Best['Month_s'][i] == 1.0:
        var = 1
        test_ny.append(var)
    elif Best['Month_s'][i] == 2.0:
        var = 2
        test_ny.append(var)
    elif Best['Month_s'][i] == 3.0:
        var = 3
        test_ny.append(var)
    elif Best['Month_s'][i] == 4.0:
        var = 4
        test_ny.append(var)
    elif Best['Month_s'][i] == 5.0:
        var = 5
        test_ny.append(var)
    elif Best['Month_s'][i] == 0.0:
        test_ny.append(var)
    
Best['Index_best'] = test_ny
Best['Index_best'] = Best['Index_best'].fillna(0)
    
#legger inn kurser, detter
conditions1  = [(Best['Index_best'] == 0),
                (Best['Index_best'] == 1),
                (Best['Index_best'] == 2),
                (Best['Index_best'] == 3),
                (Best['Index_best'] == 4),
                (Best['Index_best'] == 5),]
    
choices1 = ['UTE','XACTC25', 'SLG OMXH25', 'OBXEDNB', 'XACT OMXS30', 'TLT']
Best['ETF'] = np.select(conditions1, choices1, default=0)
valg= Best['ETF'][-1:]
valg1 = Best['ETF'][Best.index[-1]]

##Mailutsendelse
test = " ETF = {} ".format(valg1)

st.write(test)  

perf=pd.DataFrame(Ind.pct_change(),index=Ind.index)

TLT=pd.DataFrame(index=Ind.index)
TLT = yf.download('TLT', start=start, end=None, parse_dates=True)
TLT.drop(['High', 'Low','Open','Volume','Close'], axis=1, inplace=True)
TLT.columns = TLT.columns.str.replace('Adj Close', 'TLT %')
TLT['TLT %'] = TLT['TLT %'].pct_change()
TLT= TLT[start:]


perf.columns = perf.columns.str.replace('Close', '%')
perf= perf[start:]
perf['TLT %'] = TLT['TLT %']


Best= Best[start:]

#legger inn kurser, detter #Bruker bare denne når vi skal inn og ut hver måned
conditions4  = [(Best['Index_best'] == 0),
               (Best['Index_best'] == 1),
               (Best['Index_best'] == 2),
               (Best['Index_best'] == 3),
               (Best['Index_best'] == 4),
               (Best['Index_best'] == 5)]
                
choices4     = [0.0, perf['OMXC %'], perf['OMXH %'], perf['OBX %'] , perf['OMXS %'], perf['TLT %']]

Best['Kvantin1 %'] = np.select(conditions4, choices4, default=0)

mult_df = pd.DataFrame(index=Best.index)

mult_df['Portfolio Value'] = ((Best['Kvantin1 %'] + 1).cumprod()) * start_cash

st.line_chart(mult_df['Portfolio Value'])






#(benchmark can be a pandas Series or ticker)
stock = Best['Kvantin1 %']
#stock2 = Best_tick['Sum']
#OBX = Index_updated['OBX_%']
#stock2 = OMXC_string
#qs.plots.snapshot(stock2)
#qs.reports.full(stock,"SPY")
#c= qs.reports.plots(stock,'yearly returns')
#st.write(c)
#metrics = qs.reports.metrics(stock)
cagr =qs.stats.cagr(stock)*100
st.write('CAGR',cagr)

Mon =qs.stats.monthly_returns(stock)*100
st.write('Monthly returns',Mon)
#st.write('Monthly returns',Mon['EOY'])

st.bar_chart(Mon['EOY'])


if st.checkbox("Vis månedlig avkastning heatmap"):
	st.write(sns.heatmap(Mon, cmap="RdYlGn",annot=True, fmt='.1f', linewidths=.5))
	# Use Matplotlib to render seaborn
	st.pyplot()




#sns.heatmap(df, annot=True)


#_ = sns.heatmap(df, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
#plot2= plt.rcParams['figure.figsize'] = [12, 8]

#plt.title("Monthly Return Plot")
#plt.show()



