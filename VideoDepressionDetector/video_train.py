import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from scipy import stats
from imblearn.under_sampling import NearMiss
from collections import Counter
from joblib import dump


class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_


root_dir = 'D:\\Datasets\\facial_videos\\'
features_dir = 'D:\\Datasets\\original_features\\sheyang\\reading_videos\\statistics_features\\'

# Load data
aus_features = pd.read_excel(features_dir + 'reading_aus_features.xlsx', index_col=0)
gaze_features = pd.read_excel(features_dir + 'reading_gaze_features.xlsx', index_col=0)
pose_features = pd.read_excel(features_dir + 'reading_pose_features.xlsx', index_col=0)
pdm_features = pd.read_excel(features_dir + 'reading_pdm_features.xlsx', index_col=0)
eye_lm_features = pd.read_excel(features_dir + 'reading_el_features.xlsx', index_col=0)
face_lm_features = pd.read_excel(features_dir + 'reading_fl_features.xlsx', index_col=0)
features = pd.concat([aus_features, gaze_features, pose_features,pdm_features,eye_lm_features, face_lm_features], axis=1)
df_labels = pd.read_csv(root_dir + '中小学筛查数据集相关文件\\' + 'dass_senior_high_school_mts_anxiety_label.csv', index_col=[0])
infos = features.merge(df_labels, left_on='cust_id', right_on='cust_id')
infos.dropna(axis=0, how='any', inplace=True)  
infos_with_positive_labels = infos[infos.anxiety_label == 1]
infos_with_negative_labels = infos[infos.anxiety_label == 0]
labels = infos['anxiety_label'].tolist()
infos_datas_with_positive_labels = infos_with_positive_labels.drop(columns=['cust_id','焦虑_score','anxiety_label'])
infos_datas_with_negative_labels = infos_with_negative_labels.drop(columns=['cust_id','焦虑_score','anxiety_label'])
infos.drop(columns=['cust_id','焦虑_score','anxiety_label'], inplace=True)

# Feature selection
reading_difference_features = []
for i in features_name:
    _, p  = stats.mannwhitneyu(infos_datas_with_positive_labels[i], infos_datas_with_negative_labels[i])
    if p <= .05:
        reading_difference_features.append(i)
infos = infos.loc[:, reading_difference_features]

# Prepare data
X = infos
y = labels
nm = NearMiss()
X_c, y_c = nm.fit_resample(X, y)
scaler = StandardScaler()
X_c = scaler.fit_transform(X_c)
pca = PCA(n_components=.98)
X_c = pca.fit_transform(X_c)

# Define classifiers
clfs = [
    ('SVC', SVC(C=11, probability=True)),
    ('XGBoostClassifier', XGBClassifier(n_estimators=3)),  
    ('Adaboost', AdaBoostClassifier(random_state=9, n_estimators=14)), 
    ('GaussianNB', GaussianNB()),
]  

clfs_names = ['SVC', 'XGBoostClassifier', 'Adaboost', 'GaussianNB']

# Train models
for clf_name, clf in clfs:
    bagging_clf = BaggingClassifier(base_estimator=clf, random_state=0, n_estimators=10, max_samples=1.0)
    bagging_clf.fit(X_c, y_c)
    dump(bagging_clf, './model/video_model.joblib')