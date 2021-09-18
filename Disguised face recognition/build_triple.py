import os

train=open('train1.txt','w')

for i in range(0,400):
    for j in range(1,6):
        train.writelines('%04d/real/%03d.png %04d/real/ %d %d 0\n'%(i,j,i,j,i))
        train.writelines('%04d/real/%03d.png %04d/real/ %d %d 1\n'%(i,j,i,j,i))

        train.writelines('%04d/real/%03d.png %04d/fake/ %d %d 2\n'%(i,j,i,j,i))
        train.writelines('%04d/real/%03d.png %04d/fake/ %d %d 3\n'%(i,j,i,j,i))

        train.writelines('%04d/fake/%03d.png %04d/real/ %d %d 4\n'%(i,j,i,j,i))
        train.writelines('%04d/fake/%03d.png %04d/real/ %d %d 5\n'%(i,j,i,j,i))

        train.writelines('%04d/fake/%03d.png %04d/fake/ %d %d 6\n'%(i,j,i,j,i))
        train.writelines('%04d/fake/%03d.png %04d/fake/ %d %d 7\n'%(i,j,i,j,i))

train.close()


test=open('test.txt','w')
for i in range(400,500):
    for j in range(1,5):
        for j1 in range(j+1,6):
            test.writelines('%04d/real/%03d.png %04d/real/%03d.png 0\n'%(i,j,i,j1))
    for j in range(1,6):
        test.writelines('%04d/real/%03d.png %04d/fake/%03d.png 0\n'%(i,j,i,j))
    for j in range(1,5):
        test.writelines('%04d/real/%03d.png %04d/fake/%03d.png 0\n' % (i, j, i, j+1))
    test.writelines('%04d/real/001.png %04d/fake/004.png 0\n' % (i, i))
    for j in range(1,5):
        for j1 in range(j+1,6):
            test.writelines('%04d/fake/%03d.png %04d/fake/%03d.png 0\n'%(i,j,i,j1))

for i in range(400,495):
    for j in range(1,5):
        for j1 in range(j+1,6):
            test.writelines('%04d/real/%03d.png %04d/real/%03d.png 1\n'%(i,j,i+j1,j1))
            test.writelines('%04d/real/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i + j1, j1))
            test.writelines('%04d/fake/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i + j1, j1))

for i in range(495,500):
    for j in range(1,5):
        for j1 in range(j+1,6):
            test.writelines('%04d/real/%03d.png %04d/real/%03d.png 1\n' % (i, j, i-j1, j))
            test.writelines('%04d/real/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i-j1, j))
            test.writelines('%04d/fake/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i-j1, j))

test.close()





