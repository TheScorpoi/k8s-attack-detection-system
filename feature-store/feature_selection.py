import pandas as pd

class FeatureSelection():
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def select_features(self, features):
        return self.feature_store.select_features(features)
    
    def _RFE_(self):
        from sklearn.datasets import make_classification
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        X = self.data.drop('label', axis=1) 

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = self.data['label']

        model = LogisticRegression(max_iter=1000, n_jobs=-1)

        rfe = RFE(model, n_features_to_select=8)
        rfe = rfe.fit(X_scaled, y)

        coefficients = model.coef_[0]

        importance = coefficients / sum(abs(coefficients))

        importance_percentage = importance * 100

        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance (%)': importance_percentage})
        print(importance_df.sort_values(by='Importance (%)', key=abs, ascending=False))
                                                  
    
                                        
if __name__ == "__main__":
    
    data = pd.read_csv("../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/preprocessed_data.csv")
    feature_selection = FeatureSelection(data=data)
    
    feature_selection._RFE_()
    
