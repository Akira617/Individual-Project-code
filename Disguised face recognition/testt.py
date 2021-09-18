import os
import random
import numpy as np


f=open('train.txt')
data=f.readlines()
list1=[k for k in range(5)]
list2=[k for k in range(400)]
for i in data:
    isa=i.split()
    if isa[-1]=='0' or isa[-1]=='4':
        img1name=isa[0]
        list10=list1.copy()
        list10.remove(int(isa[-3]))
        img2name=isa[1]+'%03d.png'%random.choice(list10)

        list20=list2.copy()
        list20.remove(int(isa[-2]))
        img3name='%04d/real/%03d.png'%(random.choice(list20),random.choice(list10))

    if isa[-1] == '1' or isa[-1]=='5':
        img1name = isa[0]
        list10 = list1.copy()
        list10.remove(int(isa[-3]))
        img2name = isa[1] + '%03d.png' % random.choice(list10)

        list20 = list2.copy()
        list20.remove(int(isa[-2]))
        img3name = '%04d/fake/%03d.png' % (random.choice(list20), random.choice(list10))

    if isa[-1]=='2' or isa[-1]=='6':
        img1name=isa[0]
        list10=list1.copy()
        img2name=isa[1]+'%03d.png'%random.choice(list10)

        list20=list2.copy()
        list20.remove(int(isa[-2]))
        img3name='%04d/real/%03d.png'%(random.choice(list20),random.choice(list10))

    if isa[-1] == '3' or isa[-1]=='7':
        img1name = isa[0]
        list10 = list1.copy()
        img2name = isa[1] + '%03d.png' % random.choice(list10)

        list20 = list2.copy()
        list20.remove(int(isa[-2]))
        img3name = '%04d/fake/%03d.png' % (random.choice(list20), random.choice(list10))


