
# i gotta create of have py file that will call all the features and there it would also load the data and then call all the yk features there and and for each of them and then for each class then we would have sth like that that would help us have the data for eahch of the classes i guess created and i think i gotta do that and extract sth like that for each of them happening so that is actually sth that i should create 
#i'm not quite srue if that should happen in .py or a jupyter notbook exactly so that is what i am gonna do 

from rna_features import RnaFeatures
from disease_features import DiseaseFeatures
from CrossFeatures import CrossFeatures
from feature_extractor import FeatureExtractor

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
