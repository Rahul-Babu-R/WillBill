from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


train_data = pd.read_csv('subnew.csv',delimiter=',')
test_data = pd.read_csv('testnew1l.csv',delimiter=',')
vals = {'SO':1,'AT':0}
train_data['solved_status'] = train_data['solved_status'].map(vals)
print any(train_data['solved_status'])
train_data = train_data.fillna(0.5)
# train_data.loc[train_data["chances_suberror"] < 0.5, "chances_suberror"] = 0
# train_data.loc[train_data["chances_suberror"] >= 0.5, "chances_suberror"] = 1
# train_data.loc[train_data["chances_rate"] < 0.6, "chances_rate"] = 0
# train_data.loc[train_data["chances_rate"] >= 0.6, "chances_rate"] = 1
# train_data.loc[train_data["level"] < 0.6555, "level"] = 0
# train_data.loc[train_data["level"] >= 0.6555, "level"] = 1
train_data.loc[train_data["user_error"] < 0.4, "user_error"] = 0
train_data.loc[train_data["user_error"] >= 0.4, "user_error"] = 1
train_data.loc[train_data["error_count"] < 150, "error_count"] = 1
train_data.loc[train_data["error_count"] >= 150, "error_count"] = 0
train_data.loc[train_data["rate"] <= 3.9, "rate"] = 1
train_data.loc[train_data["rate"] > 3.9, "rate"] = 0
print train_data.columns
print 'Loadin Algo'
predictors = ["user_id","problem_id","chances_suberror","chances_rate","rate","skill","level","user_error","accuracy","solved_count","usolved","uattempt","type","tagval"]

y = train_data['solved_status']
y = np.array(y).astype(int)
alg = RandomForestClassifier(
    random_state=1,
    n_estimators=2750,
    n_jobs=-1
)
print 'loading score'
scores = cross_validation.cross_val_score(
    alg,
    train_data[predictors],
    y
)
print(scores.mean())
# lr = ExtraTreesClassifier(n_estimators=1000,n_jobs=-1)
# scores = cross_validation.cross_val_score(
#     lr,
#     train_data[predictors],
#     y
# )
# print(scores.mean())
test_data = test_data.fillna(0.5)
alg.fit(train_data[predictors],y)
predictions =alg.predict(test_data[predictors])
submissions = pd.DataFrame({
        "Id": test_data["Id"],
        "solved_status": predictions
    })
submissions.to_csv('sub_forest2l.csv',columns=("Id","solved_status"),index=None)
# lr.fit(train_data[predictors],y)
# predic =lr.predict(test_data[predictors])
# submissions = pd.DataFrame({
#         "Id": test_data["Id"],
#         "solved_status": predic
#     })
# submissions.to_csv('sub_lr.csv',columns=("Id","solved_status"),index=None)
# alg    = LogisticRegression(random_state=1)
#
# scores = cross_validation.cross_val_score(
#     alg,
#     train_data[predictors],
#     y
# )
# print(scores.mean())



