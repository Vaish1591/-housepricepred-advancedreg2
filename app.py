import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn
import xgboost
from numpy import asarray
from sklearn.preprocessing import RobustScaler, LabelEncoder
rb= RobustScaler()
l_encoder = LabelEncoder()

app = Flask(__name__)
model = pickle.load(open('model9.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
	
    
    
    OverallQual=request.form['OverallQual']
    
    OverallQual = rb.fit_transform(asarray(OverallQual).reshape(-1,1))
    GrLivArea=request.form['GrLivArea']
    GrLivArea = rb.fit_transform(asarray(GrLivArea).reshape(-1,1))
    GarageArea=request.form['GarageArea']
    GarageArea = rb.fit_transform(asarray(GarageArea).reshape(-1,1))
    TotalBsmtSF=request.form['TotalBsmtSF']
    TotalBsmtSF = rb.fit_transform(asarray(TotalBsmtSF).reshape(-1,1))
    stFlrSF=request.form['stFlrSF']
    stFlrSF = rb.fit_transform(asarray(stFlrSF).reshape(-1,1))
    FullBath=request.form['FullBath']
    FullBath = rb.fit_transform(asarray(FullBath).reshape(-1,1))
    YearBuilt=request.form['YearBuilt']
    YearBuilt = rb.fit_transform(asarray(YearBuilt).reshape(-1,1))
    YearRemodAdd=request.form['YearRemodAdd']
    YearRemodAdd = rb.fit_transform(asarray(YearRemodAdd).reshape(-1,1))
    TotRmsAbvGrd=request.form['TotRmsAbvGrd']
    TotRmsAbvGrd = rb.fit_transform(asarray(TotRmsAbvGrd).reshape(-1,1))
    GarageYrBlt=request.form['GarageYrBlt'] 
    a= [2003.0, 1976.0, 2001.0, 1998.0, 2000.0, 1993.0, 2004.0, 1973.0,
       1931.0, 1939.0, 1965.0, 2005.0, 1962.0, 2006.0, 1960.0, 1991.0,
       1970.0, 1967.0, 1958.0, 1930.0, 2002.0, 1968.0, 2007.0, 2008.0,
       1957.0, 1920.0, 1966.0, 1959.0, 1995.0, 1954.0, 1953.0, 'None',
       1983.0, 1977.0, 1997.0, 1985.0, 1963.0, 1981.0, 1964.0, 1999.0,
       1935.0, 1990.0, 1945.0, 1987.0, 1989.0, 1915.0, 1956.0, 1948.0,
       1974.0, 2009.0, 1950.0, 1961.0, 1921.0, 1900.0, 1979.0, 1951.0,
       1969.0, 1936.0, 1975.0, 1971.0, 1923.0, 1984.0, 1926.0, 1955.0,
       1986.0, 1988.0, 1916.0, 1932.0, 1972.0, 1918.0, 1980.0, 1924.0,
       1996.0, 1940.0, 1949.0, 1994.0, 1910.0, 1978.0, 1982.0, 1992.0,
       1925.0, 1941.0, 2010.0, 1927.0, 1947.0, 1937.0, 1942.0, 1938.0,
       1952.0, 1928.0, 1922.0, 1934.0, 1906.0, 1914.0, 1946.0, 1908.0,
       1929.0, 1933.0]
    b= [89, 62, 87, 84, 86, 79, 90, 59, 19, 27, 51, 91, 48, 92, 46, 77, 56,
       53, 44, 18, 88, 54, 93, 94, 43,  8, 52, 45, 81, 40, 39, 97, 69, 63,
       83, 71, 49, 67, 50, 85, 23, 76, 31, 73, 75,  5, 42, 34, 60, 95, 36,
       47,  9,  0, 65, 37, 55, 24, 61, 57, 11, 70, 14, 41, 72, 74,  6, 20,
       58,  7, 66, 12, 82, 28, 35, 80,  3, 64, 68, 78, 13, 29, 96, 15, 33,
       25, 30, 26, 38, 16, 10, 22,  1,  4, 32,  2, 17, 21]
    mapped= zip(a,b)
    mapped = dict(mapped)   
    GarageYrBlt = float(GarageYrBlt )
    GarageYrBlt= mapped.get(GarageYrBlt)
    GarageYrBlt = rb.fit_transform(asarray(GarageYrBlt).reshape(-1,1))

    MSZoning=request.form['MSZoning']
    if (MSZoning == 'RL'):
    	MSZoning = 3
    elif (MSZoning == 'RM'):
    	MSZoning = 4
    elif (MSZoning == 'C (all)'):
    	MSZoning = 0
    elif (MSZoning == 'FV'):
    	MSZoning = 1
    elif (MSZoning == 'RH'):
    	MSZoning = 2

    CentralAir=request.form['CentralAir']

    if (CentralAir == 'Y'):
    	CentralAir = 1
    elif (CentralAir == 'N'):
    	CentralAir = 0

    Exterior1st = request.form['Exterior1st']
    
    if (Exterior1st  == 'VinylSd'):
    	Exterior1st  = 12
    elif (Exterior1st  == 'MetalSd'):
    	Exterior1st  = 8
    elif (Exterior1st  == 'Wd Sdng'):
    	Exterior1st  = 13
    elif (Exterior1st  == 'HdBoard'):
    	Exterior1st =6
    elif (Exterior1st  == 'BrkFace'):
    	Exterior1st  = 3
    elif (Exterior1st  == 'WdShing'):
    	Exterior1st  = 14
    elif (Exterior1st  == 'CemntBd'):
    	Exterior1st  = 5
    elif (Exterior1st  == 'Plywood'):
    	Exterior1st  = 9
    elif (Exterior1st  == 'AsbShng'):
    	Exterior1st  = 0
    elif (Exterior1st  == 'Stucco'):
    	Exterior1st  = 11
    elif (Exterior1st  == 'BrkComm'):
    	Exterior1st  = 2
    elif (Exterior1st  == 'AsphShn'):
    	Exterior1st  = 1
    elif (Exterior1st  == 'Stone'):
    	Exterior1st  = 10
    elif (Exterior1st  == 'ImStucc'):
    	Exterior1st  = 7
    elif (Exterior1st  == 'CBlock'):
    	Exterior1st  = 4



    SaleCondition = request.form['SaleCondition']
    	
    if (SaleCondition  == 'Normal'):
    	SaleCondition  = 4
    elif (SaleCondition  == 'Abnorml'):
    	SaleCondition  = 0
    elif (SaleCondition  == 'Partial'):
    	SaleCondition  = 5
    elif (SaleCondition  == 'AdjLand'):
    	SaleCondition  = 1
    elif (SaleCondition  == 'Alloca'):
    	SaleCondition  = 2
    elif (SaleCondition  == 'Family'):
    	SaleCondition  = 3





    BsmtExposure=request.form['BsmtExposure']
    
    if (BsmtExposure == 'None'):
    	BsmtExposure = 0
    elif (BsmtExposure == 'No'):
    	BsmtExposure = 1
    elif (BsmtExposure == 'Mn'):
    	BsmtExposure = 2
    elif (BsmtExposure == 'Av'):
    	BsmtExposure = 3
    elif (BsmtExposure == 'Gd'):
    	BsmtExposure = 4

    BsmtFinType1=request.form['BsmtFinType1']
  
    if (BsmtFinType1 == 'None'):
    	BsmtFinType1 = 0
    elif (BsmtFinType1 == 'Unf'):
    	BsmtFinType1 = 1
    elif (BsmtFinType1 == 'LwQ'):
    	BsmtFinType1 = 2
    elif (BsmtFinType1 == 'Rec'):
    	BsmtFinType1 = 3
    elif (BsmtFinType1 == 'BLQ'):
    	BsmtFinType1 = 4
    elif (BsmtFinType1 == 'ALQ'):
    	BsmtFinType1 = 5
    elif (BsmtFinType1 == 'GLQ'):
    	BsmtFinType1 = 6

    HeatingQC=request.form['HeatingQC']
   
    if (HeatingQC == 'Po'):
    	HeatingQC = 0
    elif (HeatingQC == 'Fa'):
    	HeatingQC = 1
    elif (HeatingQC == 'TA'):
    	HeatingQC = 2
    elif (HeatingQC == 'Gd'):
    	HeatingQC = 3
    elif (HeatingQC == 'Ex'):
    	HeatingQC = 4

    Electrical=request.form['Electrical']
   
    if (Electrical == 'Mix'):
    	Electrical = 0
    elif (Electrical == 'FuseP'):
    	Electrical = 1
    elif (Electrical == 'FuseF'):
    	Electrical = 2
    elif (Electrical == 'FuseF'):
    	Electrical = 3
    elif (Electrical == 'SBrkr'):
    	Electrical = 4
    elif (Electrical == 'None'):
    	Electrical = 5

    FireplaceQu=request.form['FireplaceQu']
   
    if (FireplaceQu == 'None'):
    	FireplaceQu = 0
    elif (FireplaceQu == 'Po'):
    	FireplaceQu = 1
    elif (FireplaceQu == 'Fa'):
    	FireplaceQu = 2
    elif (FireplaceQu == 'TA'):
        FireplaceQu = 3
    elif (FireplaceQu == 'Gd'):
        FireplaceQu = 4
    elif (FireplaceQu == 'Ex'):
    	FireplaceQu = 5

    GarageCond=request.form['GarageCond']
    
    if (GarageCond == 'None'):
    	GarageCond = 0
    elif (GarageCond == 'Po'):
    	GarageCond = 1
    elif (GarageCond == 'Fa'):
    	GarageCond = 2
    elif (GarageCond == 'TA'):
        GarageCond = 3
    elif (GarageCond == 'Gd'):
    	GarageCond = 4
    elif (GarageCond == 'Ex'):
        GarageCond = 5

    PavedDrive=request.form['PavedDrive']
    
    if (PavedDrive == 'N'):
    	PavedDrive = 0
    elif (PavedDrive == 'P'):
    	PavedDrive = 1
    elif (PavedDrive == 'Y'):
    	PavedDrive = 2
 
    Fence=request.form['Fence']
    
    if (Fence == 'None'):
    	Fence = 0
    elif (Fence == 'MnWw'):
    	Fence = 1
    elif (Fence == 'GdWo'):
    	Fence =2 
    elif (Fence == 'MnPrv'):
    	Fence = 3
    elif (Fence == 'GdPrv'):
    	Fence = 4

    NbHd_num= request.form['NbHd_num']
    nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
    nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']
    if NbHd_num in nbhd_catg3:
    	NbHd_num = 2
    elif NbHd_num in nbhd_catg2:
    	NbHd_num = 3
    lits = [OverallQual, GrLivArea, GarageArea, TotalBsmtSF, stFlrSF,
       FullBath, YearBuilt, YearRemodAdd, GarageYrBlt, MSZoning,
       Exterior1st, BsmtExposure, BsmtFinType1, HeatingQC,
       CentralAir, Electrical, FireplaceQu, GarageCond, PavedDrive,
       Fence, TotRmsAbvGrd, SaleCondition,NbHd_num]
    float_features = [float(x) for x in lits]
    final_features = np.array([float_features])
    
    prediction1 = model.predict(final_features)

    prediction= np.exp(prediction1)
    output = round(prediction[0], 5)

    return render_template('index.html', prediction_text='House price estimated should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

