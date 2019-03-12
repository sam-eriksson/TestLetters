import torch.utils.data as dataset
from os import walk
from skimage import io

class LettersDataset(dataset.Dataset):
    
    def __init__(self, root_dir=".", transform=None, train=True):
        print("init")
        self.root_dir  = root_dir
        self.transform = transform
        self.train = train
        self.load()
        #self.totensor = tt.ToTensor
        
    def load(self):
        print("load")
        
        mypath  = self.root_dir
        if (self.train):
            mypath = mypath + "/train"
        else:
            mypath = mypath +"/test"
        filenames = []
        dirs = [] 
        self.classAfilenames = {}
        for (dirpath, dirnames, fnames) in walk(mypath):
            dirs.extend(dirnames)
        for dir in dirs:
            for (dp, dnames, flenames) in walk(mypath+"/"+dir):
                for name in flenames:
                    d = {dp + "/" + name : dir }
                    self.classAfilenames.update(d)
        print(self.classAfilenames)
        self.listClass = list(self.classAfilenames)
                    
        
        
    def __len__(self):
        return self.listClass.__len__()
    
    def __getitem__(self, idx):
        #print("getitem")
        path  = self.listClass[idx]
        classOf = self.classAfilenames.get(path)       
        image = io.imread(path)
        
        if self.transform:
            im = self.transform(image)
        sample = {"image" : im, "label" : ord(classOf) - 96}
        #sample = {"image" : im, "label" : classOf}
        return sample
        
       
        
        
        
        
