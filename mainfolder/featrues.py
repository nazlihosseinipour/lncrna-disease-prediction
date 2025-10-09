# this is gonna be the parent class 





# so here we get the data and based on what it is we call the right class and on each claass we've gto the extractor right? and there we call the extractor 


#gotta double check if wrong counts the first number 
# gotta add extractor function to rna_features and corssFeatrs class

from rna_features import RnaFeatures
from disease_features import DiseaseFeatures
from CrossFeatures import CrossFeatures

class FeatureExtractor:
    def __init__(self):
        self.rna = RnaFeatures()
        self.dis = DiseaseFeatures()
        self.cross = CrossFeatures()

    def run(self, method_id, *args, **kwargs):
        if method_id in range(1,13):
            return self.rna.extract(method_id, *args, **kwargs)
        elif method_id in range(13, 16):
            return self.dis.extract(method_id, *args, **kwargs)
        elif method_id in range(16, 18):
            return self.cross.extract(method_id, *args, **kwargs)
        else:
            raise ValueError(f"Unknown method id {method_id}")
