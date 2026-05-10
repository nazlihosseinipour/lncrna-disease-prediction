import copy
class Wrapper:

    def __init__(self,
                    model):
        self.model = copy.deepcopy(model)
        self.model_T = copy.deepcopy(model)

    
    def fit(self,
                x,
                x_T,
                y):
        self.model.fit(x,
                    y)
        #x_T, y_T = self._transpose_x_y(x,
#                                    y)
        y_T = y.T
        # print(y.shape)
        print(x_T.shape)
        print(y_T.shape)
        self.model_T.fit(x_T,
                        y_T)
        
    def predict_proba(self,
                    x,
                    x_T,
                    y = None):
        predictions = self.model.predict_proba(x,
                    y)
        # x_T, y_T = self._transpose_x_y(x,
        #                             y)
        if y is not None:
            y_T = y.T
        else:
            y_T = None
        predictions_T = self.model_T.predict_proba(x_T,
                                        y_T)
        print(x.shape)
        print(x_T.shape)
        print(predictions.shape)
        print(predictions_T.shape)
        final_predictions = (predictions + predictions_T)/2
        return final_predictions
    def _transpose_x_y(self,
                        x,
                        y = None):
        if y is not None:
            return x.T, y.T                    
        else:
            return x.T
