import os
import pandas as pd
import random

def create_testing_set(directory, tracks=None, genre=False, numpy=True):
    testing = []

    df = None

    if not tracks == None:
        df = pd.read_csv(tracks, header=None, usecols=[0, 40, 51, 52])
        df = df.rename(columns={0: 'id', 40:'genre', 51:'tags', 52:'title'})
        df = df.drop([0, 1, 2])
        df = df[~df['tags'].isin(['[]'])]
        df['new_tags'] = (df['tags'].str.replace(']', ',')) + ' ' + df['genre'] + ']'
        df['new_tags'].fillna(df['tags'], inplace=True)
        #print(df)

    for root, dirs, files in os.walk(directory):
        path = root.split(os.sep)
        for file in files:
            f = os.path.join(root, file)
            if os.path.isfile(f):
                if '.wav' not in f:
                    continue
                label = f.replace('fma_small\\', '')
                label = label.replace('\\', '')
                label = label.replace('.wav','')
                label = int(int(label) % 1000000)
                index = df.index[df['id']==label]
                text = 'default'
                for i in index:
                    text = []
                    text.append(df.at[i, 'id'])
                    if genre:
                        text.append(df.at[i, 'genre'])
                    text.append(df.at[i, 'new_tags'])
                    text.append(df.at[i, 'title'])
                #torch.save(wav, save_path + '/' + text + '.pt')
                #music_dict.append((wav, label)
                if type(text) == str:
                    continue
                #print(text)
                testing.append(text)
    if genre:
        testing = pd.DataFrame(testing, columns=['id', 'genre', 'tags', 'title'])
    else:
        testing = pd.DataFrame(testing, columns=['id', 'tags', 'title'])
    if numpy:
        return testing.to_numpy
    else:
        return testing

if __name__ == '__main__':
    test_set = create_testing_set('fma_small', 'fma_metadata/tracks.csv', genre=True, numpy=False)
    #print(len(test_set))
    #print(test_set)
    '''
    for j in ['Deep Sky Blue', 'Track 1','Comic Topic', 'Rocket Into The Future', 'Phase Of Prisms', 'Barcelona Afrobeat 07', 'Mesin Penenun Hujan', 'Roka']:
        index = test_set.index[test_set['title'] == j]
        for i in index:
            print(j + ': ' + str(test_set.at[i, 'id']))
    '''
    filename = '128x1024_b32_results.txt'
    flag = False
    with open(filename) as f:
        for line in f.readlines():
            if flag:
                num = int(line[line.index(':')+2:line.index('k')-1])
                try:
                    print('guess: '+ str(test_set.at[num, 'id']) + ' ' + test_set.at[num, 'title'])
                except:
                    print('failed')
                flag=False
            if 'num' in line:
                continue
            if 'true' in line:
                if int(line[line.index(':')+2:line.index('k')-1]) in [0, 14, 63, 47, 52, 10, 18, 62]:
                    print(line[line.index(':')+2:line.index('k')-1])
                    flag = True

    #print(test_set.at[421, 'id'])
    #print(test_set.at[421, 'title'])