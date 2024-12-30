from string import ascii_letters
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_k_value_difference(df, model_name, train=False):
    df['k_difference'] = df['top_k_score'] - df['actual_k_score']
    print(df)
    k_diff_data = df['k_difference'].to_numpy()
    plt.hist(k_diff_data, bins=10)
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.xlabel('Difference in k values')
    plt.ylabel('Frequency')
    if train:
        plt.title(model_name + ' Train Results - k value Histogram')
    else:
        plt.title(model_name + ' Test Results - k value Histogram')
    plt.show()

def get_num_correct(df):
    df['correct'] = (df['guess_index'] == df['actual_index'])
    print(df['correct'].value_counts())

def make_plot(items, model_name, actual_guess):
    #items = items.to_numpy()
    plt.hist(items)
    plt.xlabel('k values')
    plt.ylabel('frequency')
    plt.title(model_name + ' Test Results - k value Histogram for: ' + actual_guess)
    plt.show()

def read_results(file_name, model_name, exit_num):
    counter = 0
    num_counter = 0
    row = []
    all_rows = []

    with open(filename, 'r') as file:
        for line in file:
            string = line[line.index(':')+2:]
            string = string.strip()
            if counter == 0:
                # top k score
                row.append(float(string))
                counter += 1
            elif counter == 1:
                # guess index
                row.append(int(string))
                counter += 1
            elif counter == 2:
                # actual similarity
                row.append(float(string))
                counter += 1
            elif counter == 3:
                # true index
                row.append(int(string))
                counter += 1
            elif counter == 4:
                # true title
                row.append(string)
                all_rows.append(row)
                row = []
                counter = 0
            num_counter += 1
            if num_counter == exit_num*5:
                break
    df = pd.DataFrame(all_rows, columns=['top_k_score', 'guess_index', 'actual_k_score', 'actual_index', 'actual_title'])
    get_num_correct(df)

def read_training_results(filename, model_name, exit_num):
    counter = 0
    num_counter = 0
    row = []
    all_rows = []

    with open(filename, 'r') as file:
        for line in file:
            string = line[line.index(':')+2:]
            string = string.strip()
            if counter == 0 or counter == 1:
                # placement of true song | k value
                # placement of guess song | k value 
                placement = string[0:string.index('k')]
                k_value = string[string.index(':')+2:]
                row.append(int(placement))
                row.append(float(k_value))
                counter += 1
            elif counter == 2:
                # num true
                row.append(int(string))
                counter += 1
                all_rows.append(row)
                row = []
                counter = 0
            num_counter += 1
            if num_counter == exit_num*5:
                break
    df = pd.DataFrame(all_rows, columns=['actual_index', 'actual_k_score', 'guess_index', 'top_k_score', 'num_true'])
    print(df['num_true'].value_counts())
    get_k_value_difference(df, model_name, train=True)


if __name__ == '__main__':
    model = '128x1024 b32'
    model_name = model.replace(" ", "_")
    filename = model_name + '_results.txt'
    exit_num = 100
    
    read_training_results(filename, model_name, exit_num)