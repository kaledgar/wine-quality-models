import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score

df_red = pd.read_csv('red_wine_data.csv')
df_white = pd.read_csv('white_wine_data.csv')

# we are analyzing two different, but similar datasets
# it's convenient to create a class Pipeline - it can help us to automate data preprocessing (scaling and split)
# and model evaluation


class Pipeline:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.x = dataframe.drop(['quality', 'label'], axis = 1)
        self.y = dataframe['label']

    def _split(self, test_size = .2):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=test_size,random_state=99)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def _normalization(self):
        mm_scaler = MinMaxScaler()
        fit = mm_scaler.fit(self.x_train)
        self.x_train = fit.transform(self.x_train)
        self.x_test = fit.transform(self.x_test)

    def _process_data(self):
        self._split()
        self._normalization()

    def evaluate_and_test_models(self):
        self._process_data()


        models = [
            RandomForestClassifier(),
            DecisionTreeClassifier(),
            LogisticRegression(),
            SVC(kernel='rbf')
        ]

        model_evaluation_stats_dictionary = {
            'model': [],
            'f1-score': [],
            'precision': [],
            'recall': []
        }
        self.evaluation_statistics = pd.DataFrame(model_evaluation_stats_dictionary)

        for model in models:
            # model evaluation
            model.fit(self.x_train, self.y_train)
            y_predicted = model.predict(self.x_test)

            # score
            # precision = tp/(tp + fp), where tp, fp - true positives, false positives
            # recall = tp/(tp + fn), where fn - false positives

            score = model.score(self.x_test, self.y_test)
            score1 = accuracy_score(self.y_test, y_predicted)
            precision = precision_score(self.y_test, y_predicted)
            recall = recall_score(self.y_test, y_predicted)
            report = classification_report(self.y_test, y_predicted)
            new_row = pd.DataFrame({
                'model': [str(model)],
                'f1-score': [score1.round(2)],
                'precision': [precision.round(2)],
                'recall': [recall.round(2)]
            })
            self.evaluation_statistics = pd.concat([self.evaluation_statistics, new_row], ignore_index=True)
            #print(report)
        return self.evaluation_statistics

rw = Pipeline(df_red)
stats = rw.evaluate_and_test_models()

print(stats)
