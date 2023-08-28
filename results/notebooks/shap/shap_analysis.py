import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

sys.path.append('../../helper')
from preprocess import transform_attributes


class FeatureAnalysis(object):
    def __init__(self, metrics, transform=True,
                 keyword=None, drop_scale=True, drop_step=False):
        self.metrics = metrics
        x_vars_prelim = [
            'Algo', 'Preset', 'Lr', 'Dim', 'Alpha', 'Factor'
        ]
        if not drop_step:
            x_vars_prelim.append('Step')
        X_aux = metrics[x_vars_prelim].copy()
        X_aux.columns = X_aux.columns.droplevel([1, 2, 3])
        if transform:
            self.X = transform_attributes(X_aux, drop_scale=drop_scale)
            # Rows where Size == 1
            rows_to_repeat = self.X[self.X['Capacity'] == 1]
            # Append these rows twice to the original DataFrame
            self.X_new = pd.concat([self.X, rows_to_repeat], ignore_index=True)
        else:
            self.X = X_aux
        self.x_vars = self.X.columns
        self.cat_features = ['Algo', 'Preset']
        self.keyword = keyword
        self.X_for_plot = self.transform_X_for_plot(self.X)

    def transform_X_for_plot(self, X):
        plot_orders = {
            'Algo': ['lora', 'loha', 'lokr'],
            'Preset': ['attn-only', 'attn-mlp', 'full'],
        }
        df = X.copy()
        # Replacing the values
        for col in df.columns:
            if col in plot_orders:
                df[col] = df[col].apply(lambda x: plot_orders[col].index(x)
                                        if x in plot_orders[col] else x)
        for col in df.columns:
            df[col] = df[col].astype('float64')
        return df

    def fit(self, metric_names, test_size, random_state=5, **kwargs):
        if not isinstance(metric_names, list):
            metric_names = [metric_names]
        self.models = {}
        for metric_name in metric_names:
            print(f'Fitting {metric_name}')
            model = self.fit_single(metric_name, test_size, random_state,
                                    **kwargs)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X)
            self.models[metric_name] = {
                'model': model,
                'explainer': explainer,
                'shap_values': shap_values,
            }

    def fit_single(self,
                   metric_name,
                   test_size=None,
                   random_state=5,
                   **kwargs):
        y = np.array(self.metrics[metric_name + ('mean', )])
        if test_size is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, y_train = self.X, y
        model = CatBoostRegressor(iterations=300,
                                  learning_rate=0.1,
                                  random_seed=123,
                                  nan_mode='Min')
        model.fit(X_train,
                  y_train,
                  cat_features=self.cat_features,
                  verbose=False,
                  plot=False)

        pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        r2 = r2_score(y_train, pred)
        print('Training performance')
        print('RMSE: {:.2f}'.format(rmse))
        print('R2: {:.2f}'.format(r2))
        if test_size is not None:
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            print('Testing performance')
            print('RMSE: {:.2f}'.format(rmse))
            print('R2: {:.2f}'.format(r2))
        return model

    def plot_importance(self, metric_name, save=False, save_name=None):
        model = self.models[metric_name]['model']
        sorted_feature_importance = model.feature_importances_.argsort()
        plt.barh(self.x_vars[sorted_feature_importance],
                 model.feature_importances_[sorted_feature_importance],
                 color='turquoise')
        title = f"{self.keyword}_{metric_name}_feature_importance"
        plt.xlabel(f"CatBoost Feature Importance for {metric_name}")
        plt.title(title)
        if save:
            if save_name is None:
                save_name = os.path.join(
                    f"figures/{title}.png")
            plt.savefig(save_name)
