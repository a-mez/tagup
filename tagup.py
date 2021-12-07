#!/usr/bin/env python3 
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

#Connect to the database
conn = sqlite3.connect('exampleco_db.db')
cur = conn.cursor()

#I some exploratory data analysis in the console:
#SELECT name FROM sqlite_master WHERE type = 'table';
#PRAGMA table_info(static_data)
#etc
#The tables are static_data, feat_0, ..., feat_3
#The static data has columns machine_id, install_date, model, room
#The features have columns timestamp, machine, value
#There are 20 machines, 3000 timestamps, and 60000 datapoints per feature

#We'll import the data into Python
#Arbitrary choice to use feat_0 for headers, possible because each feature's
  #headers are the same
headers = [i[0] for i in cur.execute("SELECT * FROM feat_0").description]
feat_0 = np.array(cur.fetchall())

cur.execute("SELECT * FROM feat_1")
feat_1 = np.array(cur.fetchall())

cur.execute("SELECT * FROM feat_2")
feat_2 = np.array(cur.fetchall())

cur.execute("SELECT * FROM feat_3")
feat_3 = np.array(cur.fetchall())

static_headers = [i[0] for i in cur.execute("SELECT * FROM static_data").description]
#Using a list instead of a NumPy array so that we can deal with timestamp data
  #more easily
static_data = cur.fetchall()

#We're done with the .db file, so we can close the connection
cur.close()
conn.close()

#We'll build a DataArray for each feature
#The first step is to make a list of timestamps and machines
timestamps = [pd.Timestamp(str(t)) for t in np.unique(feat_0[:,0])]

#np.unique() alphabetizes the list so we can't use it for the machines
#Luckily this list is easy to generate manually
machines = ['machine_' + str(i) for i in range(20)]

#Now let's write a function to turn the data into an array with machines and
  #timestamps as coordinates
#We'll take advantage of the fact that the data is ordered by machine then
  #timestamp
#Since there are only 60000 datapoints per feature it won't take too long to
  #just iterate through them. For a larger dataset it might be worth it to
  #research if there are more efficient algorithms for this
def make_array(data):
  result = np.zeros([3000, 20])
  data = data[:,2].astype(np.float)
  
  for i in range(3000): #timestamp
    for j in range(20): #machine
      result[i,j] = data[i + (j * 3000)]
  return result
    
feat_0 = make_array(feat_0)
feat_1 = make_array(feat_1)
feat_2 = make_array(feat_2)
feat_3 = make_array(feat_3)

#Now we need to deal with outliers
#The obvious approach is to calculate the mean and standard devation and then
  #drop the datapoints that are far away from the mean. However, since data
  #values become very small once the machines fail, they could drag down the
  #standard deviation and cause us to throw out some datapoints that are not
  #actually outliers.
#Instead, I've decided to use a local rolling mean and standard deviation. To
  #do this I've written rolling mean and std functions that return the mean at
  #a certain timestamp using the n timestamps on each side of that datapoint.
#I'm only using the data from one machine and not the data from every machine
  #because some machines fail before others, so data from one machine can't
  #necessarily be extrapolated to other machines
def rolling_mean(data, n, i,j):
  return np.nanmean(data[max(i - n, 0):min(i + n, 2999),j])

def rolling_std(data, n, i,j):
  return np.nanstd(data[max(i - n, 0):min(i + n, 2999),j])

#Now let's remove outliers
#I'm defining an outlier as a datapoint that is more than 3 standard deviations
  #from the mean.
#The decision to use 100 datapoints is arbitrary.
def drop_outliers(data):
  for i in range(3000):
    for j in range(20):
      if np.abs((data[i,j] - rolling_mean(data, 100, i, j)) / rolling_std(data, 100, i, j)) > 3:
        data[i,j] = np.nan

drop_outliers(feat_0)
drop_outliers(feat_1)
drop_outliers(feat_2)
drop_outliers(feat_3)

#Finally, we'll replace outliers so that there is no missing data. I have
  #chosen to only replace an outlier from one machine using data from that
  #machine. This is justified because each machine is fairly consistent with
  #itself and there are not many large swings between different values.
#I've used the nearest two points on each side in case there's a sequence
  #[nan, nan, nan] somewhere in the data, since otherwise np.nanmean would
  #return nan. I've weighted the points so that nearer points have a larger
  #effect than farther points
#It's sometimes recommended to replace outliers with the median instead of the
  #mean since the median is less effected by outliers. However, since the
  #outliers have already been removed, that doesn't matter here.
def replace_outliers(data):
  for i in range(3000):
    for j in range(20):
      if np.isnan(data[i,j]):
        data[i,j] = np.nanmean([data[max(0, i - 2),j],
                                data[max(0, i - 1),j],
                                data[max(0, i - 1),j],
                                data[min(2999, i + 1),j],
                                data[min(2999, i + 1),j],
                                data[min(2999, i + 2),j]])

replace_outliers(feat_0)
replace_outliers(feat_1)
replace_outliers(feat_2)
replace_outliers(feat_3)

#Now we'll make the DataArrays
feat_0 = xr.DataArray(feat_0,
                      coords=[timestamps,machines],
                      dims=[headers[0],headers[1]])
feat_1 = xr.DataArray(feat_1,
                      coords=[timestamps,machines],
                      dims=[headers[0],headers[1]])
feat_2 = xr.DataArray(feat_2,
                      coords=[timestamps,machines],
                      dims=[headers[0],headers[1]])
feat_3 = xr.DataArray(feat_3,
                      coords=[timestamps,machines],
                      dims=[headers[0],headers[1]])

#Finally we'll combine these into a Dataset
ds = xr.Dataset({0 : feat_0,
                 1 : feat_1,
                 2 : feat_2,
                 3 : feat_3})

#Now onto the bonus questions
#Let's deal with the static data
#We'll start by converting the dates into pandas Timestamps
for i in range(20):
  static_data[i] = (static_data[i][0],
                    pd.Timestamp(static_data[i][1]),
                    static_data[i][2],
                    static_data[i][3])

static_data = np.array(static_data)

#An xarray Dataset is maybe overkill for this; we could accomplish what we want
  #with a dict, but I want to keep this compatible with the rest of the ML
  #pipeline
#We'll start by making a DataArray for each variable
install_date = xr.DataArray(static_data[:,1],
                            coords=[static_data[:,0]],
                            dims=static_headers[0])
model = xr.DataArray(static_data[:,2],
                     coords=[static_data[:,0]],
                     dims=static_headers[0])
room = xr.DataArray(static_data[:,3],
                    coords=[static_data[:,0]],
                    dims=static_headers[0])

#Now we'll build the Dataset
static_data = xr.Dataset({static_headers[1] : install_date,
                          static_headers[2] : model,
                          static_headers[3] : room})

#Finally, let's analyze the data
#We'll start by finding the point where each machine fails
#Since every feature stops working at the same time, we only need to worry
  #about feat_0
#I'll define the beginning of failure when the rolling standard deviation,
  #calculated with 100 datapoints, has risen or fallen for 10 timestamps
  #straight. I chose this number because the standard deviation has dramatic
  #changes at the beginning of failure, and I chose to wait until 10 timestamps
  #because, since I'm using the rolling standard deviation, the first signs of
  #failure will show up before the machine actually fails
#I'll define the end of failure when every remaining value is less than 1
#Both of these could almost certainly be improved using machine learning, but
  #since this assignment isn't supposed to take more than 4 hours and time
  #constraints and deadlines are important in a production setting, I've
  #decided not to spend time experimenting with ML models here
def start_failure(j):
  n = 0
  inc = True
  for i in range(3000):
    if n == 10:
      return i
    elif inc and rolling_std(ds[0], 100, i + 1, j) >= rolling_std(ds[0], 100, i, j):
      n += 1
    elif not inc and rolling_std(ds[0], 100, i + 1, j) <= rolling_std(ds[0], 100, i, j):
      n += 1
    else:
      n = 0
      inc = not inc

def end_failure(j):
  for i in range(3000):
    if np.nanmax(np.abs(ds.sel(machine = 'machine_' + str(j))[0][i:])) < 1:
      return i

#Now we'll add the failure times to static_data
failure_start_times = []
failure_end_times = []

for i in range(20):
  failure_start_times.append(timestamps[start_failure(i)])
  failure_end_times.append(timestamps[end_failure(i)])

failure_start_times = np.array(failure_start_times)
failure_end_times = np.array(failure_end_times)

failure_start_times = xr.DataArray(failure_start_times,
                                   coords=[np.array(machines)],
                                   dims=static_headers[0])
failure_end_times = xr.DataArray(failure_end_times,
                                 coords=[np.array(machines)],
                                 dims=static_headers[0])

static_data['failure_start_time'] = failure_start_times
static_data['failure_end_time'] = failure_end_times

#Now let's plot the data
#plt.xticks(rotation=45, ha="right")
fig0, ax0 = plt.subplots()
ax0.set_title('feat_0')
ax0.set_xlabel('Time')
#I don't know what the y axis is measuring
#In an actual production environment I'd find that out so that I could label it
ax0.plot(timestamps, ds[0])
for tick in ax0.get_xticklabels():
  tick.set_rotation(45)


fig1, ax1 = plt.subplots()
ax1.set_title('feat_1')
ax1.set_xlabel('Time')
ax1.plot(timestamps, ds[1])
for tick in ax1.get_xticklabels():
  tick.set_rotation(45)


fig2, ax2 = plt.subplots()
ax2.set_title('feat_2')
ax2.set_xlabel('Time')
ax2.plot(timestamps, ds[2])
for tick in ax2.get_xticklabels():
  tick.set_rotation(45)


fig3, ax3 = plt.subplots()
ax3.set_title('feat_3')
ax3.set_xlabel('Time')
ax3.plot(timestamps, ds[3])
for tick in ax3.get_xticklabels():
  tick.set_rotation(45)

#Finally let's find the summary statistics for each machine
#Each array contains the std or mean for each feature and each machine
#The where statements are the xarray equivalents of SELECT WHERE statements in
  #SQL for choosing the values for each machine before, during, and after
  #failure
std_before_failure = np.zeros([4, 20])
std_during_failure = np.zeros([4, 20])
std_after_failure = np.zeros([4, 20])

mean_before_failure = np.zeros([4, 20])
mean_during_failure = np.zeros([4, 20])
mean_after_failure = np.zeros([4, 20])

for i in range(4):
  for j in range(20):
    std_before_failure[i,j] = ds[i].sel(machine='machine_' + str(j)).where(ds[i].timestamp < static_data['failure_start_time'].sel(machine_id = 'machine_' + str(j))).std(skipna=True, ddof=1)

for i in range(4):
  for j in range(20):
    std_during_failure[i,j] = ds[i].sel(machine='machine_' + str(j)).where(ds[i].timestamp >= static_data['failure_start_time'].sel(machine_id = 'machine_' + str(j))).where(ds[i].timestamp < static_data['failure_end_time'].sel(machine_id = 'machine_' + str(j))).std(skipna=True, ddof=1)

for i in range(4):
  for j in range(20):
    std_after_failure[i,j] = ds[i].sel(machine='machine_' + str(j)).where(ds[i].timestamp >= static_data['failure_end_time'].sel(machine_id = 'machine_' + str(j))).std(skipna=True, ddof=1)

for i in range(4):
  for j in range(20):
    mean_before_failure[i,j] = ds[i].sel(machine='machine_' + str(j)).where(ds[i].timestamp < static_data['failure_start_time'].sel(machine_id = 'machine_' + str(j))).mean(skipna=True)

for i in range(4):
  for j in range(20):
    mean_during_failure[i,j] = ds[i].sel(machine='machine_' + str(j)).where(ds[i].timestamp >= static_data['failure_start_time'].sel(machine_id = 'machine_' + str(j))).where(ds[i].timestamp < static_data['failure_end_time'].sel(machine_id = 'machine_' + str(j))).mean(skipna=True)

for i in range(4):
  for j in range(20):
    mean_after_failure[i,j] = ds[i].sel(machine='machine_' + str(j)).where(ds[i].timestamp >= static_data['failure_end_time'].sel(machine_id = 'machine_' + str(j))).mean(skipna=True)