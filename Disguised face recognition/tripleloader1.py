from PIL import Image
import os
import os.path
import random
import torch.utils.data

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, file,file1, transform=None,
                 loader=default_image_loader):

        self.file_name = open(file)
        self.file1=file1
        self.file_list=self.file_name.readlines()

        self.list1 = [k for k in range(5)]
        self.list2 = [k for k in range(0,400)]

        self.transform = transform
        self.loader = loader
        self.len=len(self.file_list)

    def __getitem__(self, index):
        i=self.file_list[index]
        isa=i.split()

        if isa[-1] == '0' or isa[-1] == '4':
            img1name = isa[0]
            list10 = [k for k in range(1,6)]
            list10.remove(int(isa[-3]))
            img2name = isa[1] + '%03d.png' % random.choice(list10)

            list20 = [k for k in range(0,400)]
            list20.remove(int(isa[-2]))
            img3name = '%04d/real/%03d.png' % (random.choice(list20), random.choice(list10))

        if isa[-1] == '1' or isa[-1] == '5':
            img1name = isa[0]
            list10 = [k for k in range(1,6)]
            list10.remove(int(isa[-3]))
            img2name = isa[1] + '%03d.png' % random.choice(list10)

            list20 = [k for k in range(0,400)]
            list20.remove(int(isa[-2]))
            img3name = '%04d/fake/%03d.png' % (random.choice(list20), random.choice(list10))

        if isa[-1] == '2' or isa[-1] == '6':
            img1name = isa[0]
            list10 = [k for k in range(1,6)]
            img2name = isa[1] + '%03d.png' % random.choice(list10)

            list20 = [k for k in range(0,400)]
            list20.remove(int(isa[-2]))
            img3name = '%04d/real/%03d.png' % (random.choice(list20), random.choice(list10))

        if isa[-1] == '3' or isa[-1] == '7':
            img1name = isa[0]
            list10 = [k for k in range(1,6)]
            img2name = isa[1] + '%03d.png' % random.choice(list10)

            list20 = [k for k in range(0,400)]
            list20.remove(int(isa[-2]))
            img3name = '%04d/fake/%03d.png' % (random.choice(list20), random.choice(list10))
        img1=self.loader(self.file1+img1name)
        img2 = self.loader(self.file1+img2name)
        img3 = self.loader(self.file1+img3name)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3

    def __len__(self):
        return self.len


class TestImageLoader(torch.utils.data.Dataset):
    def __init__(self, file,file1, transform=None,
                 loader=default_image_loader):

        self.file_name = open(file)
        self.file_list=self.file_name.readlines()
        self.file1=file1

        self.transform = transform
        self.loader = loader
        self.len = len(self.file_list)

    def __getitem__(self, index):
        i=self.file_list[index]
        isa=i.split()

        img1name=isa[0]
        img2name = isa[1]
        label=int(isa[2])

        img1 = self.loader(self.file1+img1name)
        img2 = self.loader(self.file1+img2name)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


    def __len__(self):
        return self.len