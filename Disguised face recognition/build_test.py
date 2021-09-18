import os



test1=open('test_ori.txt','w')
test2=open('test_fake.txt','w')
for i in range(400,500):
    for j in range(1,5):
        for j1 in range(j+1,6):
            test1.writelines('%04d/real/%03d.png %04d/real/%03d.png 0\n'%(i,j,i,j1))
    for j in range(1,6):
        test2.writelines('%04d/real/%03d.png %04d/fake/%03d.png 0\n'%(i,j,i,j))
    for j in range(1,5):
        test2.writelines('%04d/real/%03d.png %04d/fake/%03d.png 0\n' % (i, j, i, j+1))
    test2.writelines('%04d/real/001.png %04d/fake/004.png 0\n' % (i, i))
    for j in range(1,5):
        for j1 in range(j+1,6):
            test2.writelines('%04d/fake/%03d.png %04d/fake/%03d.png 0\n'%(i,j,i,j1))

for i in range(400,495):
    for j in range(1,5):
        for j1 in range(j+1,6):
            test1.writelines('%04d/real/%03d.png %04d/real/%03d.png 1\n'%(i,j,i+j1,j1))
            test2.writelines('%04d/real/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i + j1, j1))
            test2.writelines('%04d/fake/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i + j1, j1))

for i in range(495,500):
    for j in range(1,5):
        for j1 in range(j+1,6):
            test1.writelines('%04d/real/%03d.png %04d/real/%03d.png 1\n' % (i, j, i-j1, j))
            test2.writelines('%04d/real/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i-j1, j))
            test2.writelines('%04d/fake/%03d.png %04d/fake/%03d.png 1\n' % (i, j, i-j1, j))

test1.close()
test2.close()




