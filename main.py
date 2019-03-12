import Model.LettersDataset as ds
from Trainer.Training import Training

def main():
    print("main")
    t = Training()
    t.train()
    t.test()
    
main()