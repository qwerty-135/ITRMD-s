
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import pandas as pd
import datetime as dt
import math
import matplotlib.patches as mpatches
import os

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return directory_path
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/result/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory

def draw_liner(data:pd.DataFrame, label_x: str, label_y: str, ts_abnormal: list, save_path, figsize=(20, 8), is_median=True, is_shown=False):
    data = data.sort_values(by=label_x)
    X = data[label_x].values
    Y = data[label_y].values

    X_spe = list(sorted(list(set(X))))
    Y_median = [0 for _ in range(len(X_spe))]

    curX = -1
    curIndex = -1
    median = []
    for i in range(len(X)):
        if curX == X[i]:
            median.append(Y[i])
        else:
            if curIndex != -1:
                if is_median:
                    Y_median[curIndex] = np.median(median)
                else:
                    Y_median[curIndex] = np.mean(median)
            curIndex += 1
            median = [Y[i]]
            curX = X[i]

    plt.figure(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)
    # plt.xticks(rotation=15)
    plt.xticks([])

    plt.yticks([])
    plt.tick_params(labelsize=20)
    plt.xlabel('Time', fontsize=25,weight='bold')
    plt.ylabel('KPI Value', fontsize=25,weight='bold')
    ax = plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    Y_max = np.max(Y_median)
    dates = [dt.datetime.fromtimestamp(ts) for ts in ts_abnormal]
    datenums_ab = md.date2num(dates)
    i=0
    for date in datenums_ab:

        if(i==0):
            plt.plot([date, date], [0, 1.2 * Y_max], c='r', lw=1, label='anomaly')

        else:

            plt.plot([date, date], [0, 1.2 * Y_max], c='r', lw=1)
        i=i+1
    plt.legend(loc='best', fontsize=25)




    ystd = np.std(Y_median)
    ymean = np.mean(Y_median)
    threshold1 = ymean - 3 * ystd
    threshold2 = ymean + 3 * ystd
    plt.axhline(y=threshold2, c="g", ls="--", lw=2,label='threshold')

    plt.legend(loc='best', fontsize=25)
    # patch2 = mpatches.Patch(color='g', label='threshold')





    dates = [dt.datetime.fromtimestamp(ts) for ts in X_spe]
    datenums = md.date2num(dates)
    outlier = []  # 将异常值保存
    outlier_x = []

    for i in range(0, len(Y_median)):
        if (Y_median[i] < threshold1) | (Y_median[i] > threshold2):
            outlier.append(Y_median[i])
            # outlier_x.append(data_x[i])
            outlier_x.append(datenums[i])
    plt.plot(outlier_x, outlier, 'ro',lw=3.5,label="outliers")

    plt.legend(loc='best', fontsize=25)
    # patch3 = mpatches.Patch(color='r', label='outliers')


    # plt.plot(datenums, Y_median, lw=1)
    plt.plot(datenums, Y_median, lw=2.5)


    plt.savefig(save_path)
    # if is_shown:
    # plt.show()
    plt.close("all")

def my_format(name:str):
    save = name.split(r".")
    return save[1]