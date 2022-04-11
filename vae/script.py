import re
import os
import shutil
source_file = r'D:\PhysioNet\Waveform_Spectrogram'
destination_file = r'D:\PhysioNet\PCG(Pure)'

file_directory1 = r'D:\PhysioNet\training-a'
file_directory2 = r'D:\PhysioNet\training-b'
file_directory3 = r'D:\PhysioNet\training-c'
file_directory4 = r'D:\PhysioNet\training-d'
file_directory5 = r'D:\PhysioNet\training-e'
file_directory6 = r'D:\PhysioNet\training-f'
files = [file_directory1, file_directory2, file_directory3, file_directory4, file_directory5, file_directory6]

a_normal = []
a_abnormal = []

b_normal = []
b_abnormal = []

c_normal = []
c_abnormal = []

d_normal = []
d_abnormal = []

e_normal = []
e_abnormal = []

f_normal = []
f_abnormal = []

def searching_file(source_file):
    global a_normal, a_abnormal, b_normal, b_abnormal, c_normal, c_abnormal, d_normal, d_abnormal, e_normal, e_abnormal, f_normal, f_abnormal
    for everyone in os.listdir(source_file + r'\normal'):
        if re.search('a', everyone):
            if not a_normal:
                a_normal = [everyone.split('.')[0]]
            else:
                a_normal.append([everyone.split('.')[0]])
        elif re.search('b', everyone):
            if not b_normal:
                b_normal = [everyone.split('.')[0]]
            else:
                b_normal.append([everyone.split('.')[0]])
        elif re.search('c', everyone):
            if not c_normal:
                c_normal = [everyone.split('.')[0]]
            else:
                c_normal.append([everyone.split('.')[0]])
        elif re.search('d', everyone):
            if not d_normal:
                d_normal = [everyone.split('.')[0]]
            else:
                d_normal.append([everyone.split('.')[0]])
        elif re.search('e', everyone):
            if not d_normal:
                e_normal = [everyone.split('.')[0]]
            else:
                e_normal.append([everyone.split('.')[0]])
        elif re.search('f', everyone):
            if not f_normal:
                f_normal = [everyone.split('.')[0]]
            else:
                f_normal.append([everyone.split('.')[0]])
        else:
            pass

    for everyone in os.listdir(source_file + r'\abnormal'):
        if re.search('a', everyone):
            if not a_abnormal:
                a_abnormal = [everyone.split('.')[0]]
            else:
                a_abnormal.append([everyone.split('.')[0]])
        elif re.search('b', everyone):
            if not b_abnormal:
                b_abnormal = [everyone.split('.')[0]]
            else:
                b_abnormal.append([everyone.split('.')[0]])
        elif re.search('c', everyone):
            if not c_abnormal:
                c_abnormal = [everyone.split('.')[0]]
            else:
                c_abnormal.append([everyone.split('.')[0]])
        elif re.search('d', everyone):
            if not d_abnormal:
                d_abnormal = [everyone.split('.')[0]]
            else:
                d_abnormal.append([everyone.split('.')[0]])
        elif re.search('e', everyone):
            if not e_abnormal:
                e_abnormal = [everyone.split('.')[0]]
            else:
                e_abnormal.append([everyone.split('.')[0]])
        elif re.search('f', everyone):
            if not f_abnormal:
                f_abnormal = [everyone.split('.')[0]]
            else:
                f_abnormal.append([everyone.split('.')[0]])
        else:
            pass
    print(a_normal, b_normal, c_normal)


def remove_file(file_directory):
    global a_normal, a_abnormal, b_normal, b_abnormal, c_normal, c_abnormal, d_normal, d_abnormal, e_normal, e_abnormal, f_normal, f_abnormal
    if re.search('-a', file_directory):
        for everyone in a_normal:
            for file in os.listdir(file_directory):
                if re.search(everyone+'.wav', file):
                    shutil.copyfile("oldfile", "newfile")  # oldfile和newfile都只能是文件


def main():

    searching_file(source_file)
    for file in files:
        remove_file(file)



if __name__ == '__main__':
    main()
