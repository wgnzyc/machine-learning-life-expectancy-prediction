import pandas as pd
import sys
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_rows', 11000)
np.set_printoptions(threshold=np.inf)
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,BaggingRegressor,RandomForestRegressor

gdptest = pd.read_csv(sys.argv[1],names=['Country Name','year','gdpvalue'])

def meltfile(file,valuename):
    f = pd.read_csv(file)
    aftermelt = pd.melt(f,id_vars=['Country Name'])
    docu = aftermelt.rename(columns={'variable':'year', 'value':valuename})
    return docu

life = meltfile('life expectancy by country and year.csv','lifevalue')
gdp = meltfile('GDP by country and year.csv','gdpvalue')

def impute(file,valuename):
    file[valuename] = file.groupby(['Country Name'])[valuename].fillna(method = 'ffill')
    file[valuename] = file.groupby(['Country Name'])[valuename].fillna(method = 'bfill')
    file[valuename] = file.groupby("year").transform(lambda x: x.fillna(x.mean()))

impute(gdp,'gdpvalue')
impute(life,'lifevalue')
impute(gdptest,'gdpvalue')


gl = pd.merge(life, gdp,on=['Country Name', 'year'])
gl['year'] = gl['year'].astype('int')
get_dummy = pd.get_dummies(gl['Country Name'])
frames = [get_dummy,gl]
gl = pd.concat(frames,axis=1)

match = gl.iloc[:,:-3]
match = match.drop_duplicates('Country Name')
gettestdummy = gdptest.merge(gdptest.merge(match,how = 'left',on=['Country Name'],sort = False),sort=False)
cols = list(gettestdummy.columns.values)
cols.pop(cols.index('Country Name'))
cols.pop(cols.index('year'))
cols.pop(cols.index('gdpvalue'))
gettestdummy = gettestdummy[cols+['Country Name','year','gdpvalue']]


train_x = gl.drop(['Country Name','gdpvalue','lifevalue'],axis = 1)
train_y = gl['lifevalue']

test_x = gettestdummy.drop(['Country Name', 'gdpvalue'],axis = 1)

# estimators = 10
# regressor = RandomForestRegressor(n_estimators=estimators)
# regressor.fit(train_x, train_y)
# predicted_y = regressor.predict(test_x)


# treereg = DecisionTreeRegressor(criterion='mse')
# treereg.fit(train_x, train_y)
# predtree_y = treereg.predict(test_x)
# print np.mean(sklearn.cross_validation.cross_val_score(treereg,
#                                              train_x, train_y,cv = 10,scoring = 'mean_squared_error'))
estimators = 10
treereg = ExtraTreesRegressor(n_estimators=estimators)
treereg.fit(train_x, train_y)
predtree_y = treereg.predict(test_x)
with open (sys.argv[2],'w') as f:
    for i in range(len(predtree_y)):
        f.write(str(predtree_y[i])+'\n')
