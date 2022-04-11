"""
generate sin wave
"""
from __future__ import print_function

"""
Normal/Abnormal Classification
"""
import re
import shutil

file_directory1 = r'D:\PhysioNet\training-a\REFERENCE.csv'
file_directory2 = r'D:\PhysioNet\training-b\REFERENCE.csv'
file_directory3 = r'D:\PhysioNet\training-c\REFERENCE.csv'
file_directory4 = r'D:\PhysioNet\training-d\REFERENCE.csv'
file_directory5 = r'D:\PhysioNet\training-e\REFERENCE.csv'
file_directory6 = r'D:\PhysioNet\training-f\REFERENCE.csv'
file_validation = r'D:\PhysioNet\validation\REFERENCE.csv'
files = [file_directory1, file_directory2, file_directory3, file_directory4, file_directory5, file_directory6, file_validation]
sum_normal = 0
sun_abnormal = 0
old_file_list = [r'D:\PhysioNet\training-a', r'D:\PhysioNet\training-b', r'D:\PhysioNet\training-c', r'D:\PhysioNet\training-d', r'D:\PhysioNet\training-e', r'D:\PhysioNet\training-f']
new_file = r'D:\PhysioNet\ALL'


def searching_file(file_directory, old_file):
    normal = 0
    abnormal = 0
    global sum_normal
    global sun_abnormal
    with open(file_directory) as f:
        lines = f.readlines()
        for line in lines:
            L = line.split()
            if re.search(',-1', L[0]):

                shutil.copyfile(old_file+'/'+L[0].split(',')[0]+'.wav', new_file+'/Normal/'+L[0].split(',')[0]+'.wav')
                # print('%s, normal++' % L[0])
                print(L[0].split(',')[0]+'.wav has been moved to Normal')
            elif re.search(',1', L[0]):

                # print('%s, abnormal++' % L[0])
                shutil.copyfile(old_file+'/'+L[0].split(',')[0]+'.wav', new_file+'/Abnormal/'+L[0].split(',')[0]+'.wav')
                print(L[0].split(',')[0]+'.wav has been moved to Abnormal')
    # print('There are %d normal samples in %s, and %d abnormal samples in %s.' % (normal, file_directory, abnormal, file_directory))
    # sum_normal += normal
    # sun_abnormal += abnormal


def main():
    for i in range(6):
        searching_file(files[i], old_file_list[i])

    # print('Altogether there are %d normal samples' % sum_normal)
    # print('Altogether there are %d abnormal samples' % sun_abnormal)


if __name__ == '__main__':
    main()

