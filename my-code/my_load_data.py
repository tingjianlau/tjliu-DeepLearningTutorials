#!/usr/bin/env python
# -*- coding: UTF-8 -*-

##########################################################################
#	> File Name: my_load_data.py
#	> Author: Tingjian Lau
#	> Mail: tjliu@mail.ustc.edu.cn
#	> Created Time: 2016/09/22
#	> Detail: 
#########################################################################

def load_data(path):
    data = open(path, 'r')
    prices = []
    label = []
    
    for line in data:
        val = line.split('\n')[0].split()
        prices.append(val[0:40])
        label.append(val[-1])
        print(val[0:40])


if __name__ == '__main__':
    load_data('data/tmp_test.txt')
