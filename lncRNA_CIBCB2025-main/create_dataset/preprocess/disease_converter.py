
class Converter:
    
    def __init__(self):
        self.converter_mesh_to_flat = {}
        self.converter_flat_to_mesh = {}

        self.counter = {}
        self._populate_converters()
    def get_flat(self,
                flat_disease):
        if self.is_in(flat_disease):
            self.increase_counter_converter(flat_disease)
            flat_equivalent =  self.converter_mesh_to_flat[flat_disease]
            return self._get_new_mesh_id(flat_disease)
    def get_mesh(self,
                mesh_disease
                ):
        return self.converter_flat_to_mesh[mesh_disease]
    def is_in(self,
                disease):
        return disease in self.converter_mesh_to_flat
    def get_new_columns_flat(self,
                flat_columns):
        new_columns = [self.get_mesh(c) for c in flat_columns]
        return new_columns
    def _get_new_mesh_id(self, flat_disease):
        return self.converter_mesh_to_flat[flat_disease] + "_" + str(self.counter[flat_disease])
    def _populate_converters(self):
        self.converter_mesh_to_flat["Alzheimer Disease"] = "Alzheimer's disease"
        self.converter_mesh_to_flat["Huntington Disease"] = "Huntington's disease"
        self.converter_mesh_to_flat["Parkinson Disease"] = "Parkinson's disease"
        self.converter_mesh_to_flat["Leukemia, Myeloid, Acute"] = "acute myeloid leukemia"
        self.converter_mesh_to_flat["Myocardial Infarction"] = "acute myocardial infarction"
        self.converter_mesh_to_flat["Urinary Bladder Neoplasms"] = "bladder cancer"
        self.converter_mesh_to_flat["Breast Neoplasms"] = "breast cancer"
        self.converter_mesh_to_flat["Neoplasms"] = "cancer", "lymphoma"
        self.converter_mesh_to_flat["Uterine Cervical Neoplasms"] = "cervical cancer"    
        self.converter_mesh_to_flat["Colonic Neoplasms"] = "colon cancer"
        self.converter_mesh_to_flat["Colorectal Neoplasms"] = "colorectal cancer"
        self.converter_mesh_to_flat["Diabetes Mellitus"] = "diabetes mellitus"
        self.converter_mesh_to_flat["Endometrial Neoplasms"] = "endometrial cancer"  
        self.converter_mesh_to_flat["Esophageal Neoplasms"] = "esophageal cancer"
        self.converter_mesh_to_flat["Esophageal Squamous Cell Carcinoma"] = "esophageal squamous cell carcinoma"
        self.converter_mesh_to_flat["Gallbladder Neoplasms"] = "gallbladder cancer"
        self.converter_mesh_to_flat["Stomach Neoplasms"] = "gastric cancer"
        self.converter_mesh_to_flat["Glioblastoma"] = "glioblastoma"
        self.converter_mesh_to_flat["Glioma"] = "glioma"
        self.converter_mesh_to_flat["Heart Failure"] = "heart failure"
        self.converter_mesh_to_flat["Carcinoma, Hepatocellular"] = "hepatocellular carcinoma"
        self.converter_mesh_to_flat["Telangiectasia, Hereditary Hemorrhagic"] = "hereditary haemorrhagic telangiectasia"
#        self.converter_mesh_to_flat["Disease Models, Animal, Reperfusion Injury"] = "ischaemia-reperfusion injury"
        self.converter_mesh_to_flat["Disease Models, Animal"] = "ischaemia-reperfusion injury"

        self.converter_mesh_to_flat["Kidney Neoplasms"] = "kidney cancer"
        self.converter_mesh_to_flat["Liver Neoplasms"] = "liver cancer"
        self.converter_mesh_to_flat["Adenocarcinoma of Lung"] = "lung adenocarcinoma"
        self.converter_mesh_to_flat["Lung Neoplasms"] = "lung cancer"
#        self.converter_mesh_to_flat["Neoplasms"] = "lymphoma"
        self.converter_mesh_to_flat["Melanoma"] = "melanoma"
        self.converter_mesh_to_flat["Nasopharyngeal Carcinoma"] = "nasopharyngeal carcinoma"
        self.converter_mesh_to_flat["Neuroblastoma"] = "neuroblastoma"
        self.converter_mesh_to_flat["Carcinoma, Non-Small-Cell Lung"] = "non-small cell lung cancer"
        self.converter_mesh_to_flat["Osteoarthritis"] = "osteoarthritis"
        self.converter_mesh_to_flat["Osteosarcoma"] = "osteosarcoma"
        self.converter_mesh_to_flat["Ovarian Neoplasms"] = "ovarian cancer"
        self.converter_mesh_to_flat["Pancreatic Neoplasms"] = "pancreatic cancer"
        self.converter_mesh_to_flat["Carcinoma, Pancreatic Ductal"] = "pancreatic ductal adenocarcinoma"
        self.converter_mesh_to_flat["Thyroid Cancer, Papillary"] = "papillary thyroid carcinoma"
        self.converter_mesh_to_flat["Prostatic Neoplasms"] = "prostate cancer"
        self.converter_mesh_to_flat["Carcinoma, Renal Cell"] = "renal cell carcinoma"
        self.converter_mesh_to_flat["Adenocarcinoma, Clear Cell"] = "renal clear cell carcinoma"
        self.converter_mesh_to_flat["Hallucinations"] = "schizophrenia"
        self.converter_mesh_to_flat["Thyroid Neoplasms"] = "thyroid cancer"
        #self.converter_mesh_to_flat["Carcinoma, Squamous Cell, Tongue Neoplasms"] = "tongue squamous cell carcinoma"
        self.converter_mesh_to_flat["Carcinoma, Squamous Cell"] = "tongue squamous cell carcinoma"
        self.converter_mesh_to_flat["Diabetes Mellitus, Type 2"] = "type 2 diabetes mellitus"

        self.converter_flat_to_mesh = dict((k,v) for v,k in self.converter_mesh_to_flat.items())
        self.converter_flat_to_mesh["lymphoma"] = "Neoplasms"
        
    def increase_counter_converter(self, flat_disease):
        counter_key = flat_disease
        if counter_key not in self.counter:
            self.counter[counter_key] = 1
        else:
            self.counter[counter_key] += 1
    def get_counter(self):
        return self.counter
    def get_not_found_diseases(self):
        return set(self.converter_mesh_to_flat.keys()) - set(self.counter.keys())
