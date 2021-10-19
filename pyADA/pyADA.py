import numpy as np
from numpy import log, e
from math import sqrt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, accuracy_score

class Smetrics:
    def __init__(self):
        self.maxdata = None
        self.mindata = None
    
    def rescale_norm(self, data):
        """
        Perform a normalization rescale to range [0, 1]. The maximum and minimum
        values can be accessed in Smetrics.maxdata and SMetrics.mindata respectively.
        """
        self.maxdata = max(data)
        self.mindata = min(data)
        data = np.array(data)
        sdata = ((data - min(data)) / (max(data) - min(data)))
        return sdata
        
    def back_normrescale(self, data):
        """
        Back rescale to normal data.
        """
        data = np.array(data)
        bsdata = (data*(self.maxdata - self.mindata) + self.mindata)
        return bsdata
    
    def rescale_ln(self, data):
        """
        Apply ln function in a set of data
        """
        data = np.array(data)
        return log(data)
    
    def back_lnrescale(self, data):
        """
        Back ln values to the original data.
        """
        data = np.array(data)
        return e**(data)
    
    def absolute_relative_error(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.absolute(y_true - y_pred) / y_true
    
    def mean_absolute_error(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return sum( (np.absolute(y_true - y_pred)) / (len(y_true)) )
    
    def mean_square_error(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return sum( (y_true - y_pred)**2 ) / len(y_true)
    
    def roots_mean_square_error(self, y_true, y_pred):
        return sqrt(Smetrics.mean_square_error(self, y_true=y_true, y_pred=y_pred))
    
    def Q2ext(self, y_true, y_pred, y_train):
        """
        Correlation coefficient of external validation. Also known as external explained variance
        or prediction power of model.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_train = np.array(y_train)
        q2calc = 1 - ((sum((y_true - y_pred)**2)) / (sum((y_true - np.mean(y_train))**2)))
        return q2calc
    
    def R2ext(self, y_true, y_pred):
        """
        Correlation coefficient of multiple determination. Also known as coefficient
        of multiple determination, multiple correlation coefficient and explained
        variance in fitting.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        R2calc = 1 - ((sum((y_true - y_pred)**2)) / (sum((y_true - np.mean(y_true))**2)))
        return R2calc
    
    def Q2int(self, y_true, y_pred):
        """
        Crossvalidated correlation coefficient. Also known as (LOO or LNO)
        crossvalidated correlation coefficient, explained variance in prediction,
        (LOO or LNO) crossvalidated explained variance,and explained variance by
        LOO or by LNO. The attributes LOO and LNO are frequently omitted in names
        for this correlation coefficient.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        q2int = 1 - ((sum((y_true - y_pred)**2)) / (sum((y_true - np.mean(y_true))**2)))
        return q2int

    def kappaCalc(self, y_true, y_pred):
        # Matrix confussion
        true = np.array(y_true)
        pred = np.array(y_pred)
        true = pd.Series(true, name='True')
        pred = pd.Series(pred, name='Predicted')
        df_confusion = pd.crosstab(true, pred)
        df_confusion

        # sum diagonal
        mat_confussion = np.array(df_confusion)
        diagonal_sum = mat_confussion.diagonal().sum()

        # Kappa equation
        p0 = (diagonal_sum)/len(y_true) # the relative acceptance rate
        pe = np.array([(mat_confussion[i,:].sum()*mat_confussion[:,i].sum())/(mat_confussion.sum()**2) for i in range(0,len(np.unique(y_true)))]).sum()
        kappa = (p0 - pe)/(1 - pe)

        # p-value


        return kappa


class Similarity:
    """
        All similarity calculations have a range of [0, 1]
    """
    def __init__(self):        
        pass    
    
    def __coefs(self, vetor1, vetor2):
        A = np.array(vetor1).astype(int)
        B = np.array(vetor2).astype(int)

        AnB = A & B #intersection
        onlyA = np.array(B) < np.array(A) #A is a subset of B
        onlyB = np.array(A) < np.array(B) #B is a subset of A
        AuB_0s = A | B #Union (for count de remain zeros)
        
        return AnB,onlyA,onlyB,np.count_nonzero(AuB_0s==0)
    
    def tanimoto_similarity(self, vetor1, vetor2):
        """
        Structural similarity calculation based on tanimoto index.
        T(A,B) = (A ^ B)/(A + B - A^B)
        """
        
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)
        return AnB.sum() / (onlyA.sum() + onlyB.sum() + AnB.sum())
    
    
    def tversky_similarity(self, vetor1, vetor2, alpha=1, beta=1):
        """
        The alpha and beta coefficients weight the similarity
        calculation, giving greater weight to the reference
        compound or to the query compound.
        """
        
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)
        return AnB.sum() / (alpha*onlyA.sum() + beta*onlyB.sum() + AnB.sum())
    
    def geometric_similarity(self, vetor1, vetor2):
        """
        Cosine Similarity - Calculates the ratio of the bits in common to
        the geometric mean of the number of “on” bits
        in the two fingerprints.
        """
        
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)        
        return AnB.sum() / ( sqrt((onlyA.sum() + AnB.sum()) * (onlyB.sum() + AnB.sum())) )
    
    def arithmetic_similarity(self, vetor1, vetor2):
        """
        Dice similarity - Calculates the ratio of the bits in common to the
        arithmetic mean of the number of “on” bits in the
        two fingerprints.
        """
        
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)        
        return 2*AnB.sum() / ( onlyA.sum() + onlyB.sum() + (2*AnB.sum()) )
    
    def euclidian_similarity(self, vetor1, vetor2):
        """
        Similarity based on Euclidean distance of two vectors.
        """
        
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)
        return sqrt( (AnB.sum()  + AuB_0s ) / (onlyA.sum()  + onlyB.sum()  + AnB.sum() + AuB_0s) )
   
    def manhattan_similarity(self, vetor1, vetor2):
        """
        Similarity based on Manhattan distance of two vectors.
        """
        
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)
        return (onlyA.sum() + onlyB.sum()) / (onlyA.sum() + onlyB.sum() + AnB.sum() + AuB_0s)

class ApplicabilityDomain:
    def __init__(self, verbose=False):
        self.__sims = Similarity()    
        self.__smetrics = Smetrics()
        self.__verbose=verbose
        self.similarities_table_ = None
                
    def analyze_similarity(self, base_test, base_train, similarity_metric='tanimoto', alpha=1, beta=1):
        """
        Analysis of the similarity between molecular fingerprints
        using different metrics. A table (dataframe pandas) will be
        generated with the coefficients: Average, median, std,
        maximum similarity and minimum similarity, for all compounds
        in the test database in relation to the training database.
        The alpha and beta parameters are only for 'tversky' metric.
        
        Arguments
        ---------------------
        base_test
            Must be a numpy matrix containing only the fingerprints of all molecules in the test database.
            
                >>> Example of data format
                array([[0., 0., 0., ..., 0., 1., 0.],
                       [0., 0., 0., ..., 0., 0., 0.],
                       [0., 0., 1., ..., 0., 0., 0.],
                       ...,
                       [1., 0., 0., ..., 0., 0., 1.],
                       [0., 1., 1., ..., 1., 0., 0.],
                       [0., 0., 1., ..., 0., 1., 1.]])
        base_train
            Must be a numpy matrix containing only the fingerprints of all molecules in the training database.
            
                >>> Example of data format
                array([[0., 0., 0., ..., 0., 1., 0.],
                       [0., 0., 0., ..., 0., 0., 0.],
                       [0., 0., 1., ..., 0., 0., 0.],
                       ...,
                       [1., 0., 0., ..., 0., 0., 1.],
                       [0., 1., 1., ..., 1., 0., 0.],
                       [0., 0., 1., ..., 0., 1., 1.]])
        similarity_metric
            Desired metric to perform the similarity calculation. The options are:
            'tanimoto', 'tversky', 'geometric', 'arithmetic', 'euclidian', 'manhattan'.
            The default metric is tanimoto index.
            
                >>> Example of use
                ad = ApplicabilityDomain()
                ad.analyze_similarity(base_test, base_train, similarity_metric='euclidian')
                or
                ad.analyze_similarity(base_test, base_train, similarity_metric='tversky', alpha=1, beta=1)
            
            The alpha and beta parameters are only for 'tversky' metric.
        """
               
        get_tests_similarities = []
        similarities = {}
        #get dictionary of all data tests similarities
        if self.__verbose:
            with tqdm(total=len(base_test)) as progbar:
                for n,i_test in enumerate(base_test):
                    for i_train in base_train:
                        if similarity_metric == 'tanimoto':
                            get_tests_similarities.append(self.__sims.tanimoto_similarity(i_test, i_train))
                        elif similarity_metric == 'tversky':
                            get_tests_similarities.append(self.__sims.tversky_similarity(i_test, i_train, alpha=alpha, beta=beta))
                        elif similarity_metric == 'geometric':
                            get_tests_similarities.append(self.__sims.geometric_similarity(i_test, i_train))
                        elif similarity_metric == 'arithmetic':
                            get_tests_similarities.append(self.__sims.arithmetic_similarity(i_test, i_train))
                        elif similarity_metric == 'manhattan':
                            get_tests_similarities.append(self.__sims.manhattan_similarity(i_test, i_train))
                        elif similarity_metric == 'euclidian':
                            get_tests_similarities.append(self.__sims.euclidian_similarity(i_test, i_train))                    
                        else:
                            get_tests_similarities.append(self.__sims.tanimoto_similarity(i_test, i_train))
                        
                    similarities['Sample_test_{}'.format(n)] = np.array(get_tests_similarities)
                    get_tests_similarities = []
                    progbar.update(1)
        else:
            for n,i_test in enumerate(base_test):            
                for i_train in base_train:
                    if similarity_metric == 'tanimoto':
                        get_tests_similarities.append(self.__sims.tanimoto_similarity(i_test, i_train))
                    elif similarity_metric == 'tversky':
                        get_tests_similarities.append(self.__sims.tversky_similarity(i_test, i_train, alpha=alpha, beta=beta))
                    elif similarity_metric == 'geometric':
                        get_tests_similarities.append(self.__sims.geometric_similarity(i_test, i_train))
                    elif similarity_metric == 'arithmetic':
                        get_tests_similarities.append(self.__sims.arithmetic_similarity(i_test, i_train))
                    elif similarity_metric == 'manhattan':
                        get_tests_similarities.append(self.__sims.manhattan_similarity(i_test, i_train))
                    elif similarity_metric == 'euclidian':
                        get_tests_similarities.append(self.__sims.euclidian_similarity(i_test, i_train))                    
                    else:
                        get_tests_similarities.append(self.__sims.tanimoto_similarity(i_test, i_train))
                    
                similarities['Sample_test_{}'.format(n)] = np.array(get_tests_similarities)
                get_tests_similarities = []
                    
        self.similarities_table_ = pd.DataFrame(similarities)
        
        
        analyze = pd.concat([self.similarities_table_.mean(),
                             self.similarities_table_.median(),
                             self.similarities_table_.std(),
                             self.similarities_table_.max(),
                             self.similarities_table_.min()],
                             axis=1)        
        analyze.columns = ['Mean', 'Median', 'Std', 'Max', 'Min']
        
        return analyze
            
    
    def fit(self, model, base_test, base_train, y_true, isTensorflow=False,
            threshold_reference = 'max', threshold_step = (0, 1, 0.05),
            similarity_metric='tanimoto', alpha = 1, beta = 1, metric_avaliation='rmse'):
        
        
        """
        Performs a scan of the similarities between the compounds
        in the test set in relation to the training set. This scan
        is done in terms of the "threshold", for each threshold an
        error value (rmse, mse or mae) and a vector containing the
        positions of the fingerprints are associated, these fingerprints
        have a similarity value lower than the threshold.
        This function returns a dictionary where the key corresponds
        to a string (threshold value), and the value corresponds to a
        list where the first position refers to the error value and the
        second position corresponds to the positions (index) of the
        fingerprints in the test-set <= threshold. To perform the
        error/correlation calculation,
        it is necessary to predict the values of the class (y_pred),
        so this function interacts with the models created in the
        scikit-learn and xgboost packages. Futures packages of ML
        that perform like scikit-learn maybe works.
        
        Example of use
            >>> rlr = RandomForestRegressor(n_estimators=150, n_jobs=5) #create sklearn model
            >>> rlr.fit(xi_train, y_train) #training model
            >>> AD = ApplicabilityDomain()
            >>> AD.fit(model=rlr, base_test=xi_test, base_train=xi_train, y_true=y_test,
                  threshold_step=(0, 0.035, 0.005), threshold_reference='average',
                  metric_error='rmse')
            {'Threshold 0.0': [[5212831353.552006], array([], dtype=int64)],
             'Threshold 0.005': [[5231287044.044762], array([125])],
             'Threshold 0.01': [[5231287044.044762], array([125])],
             'Threshold 0.015': [[5231287044.044762], array([125])],
             'Threshold 0.02': [[5231287044.044762], array([125])],
             'Threshold 0.025': [[5208218469.123184], array([ 99, 100, 125])],
             'Threshold 0.03': [[5208218469.123184], array([ 99, 100, 125])],
             'Threshold 0.035': [[5225313754.544512], array([  0,  99, 100, 125])]}
            
        Arguments
        ---------------------
        model
            Model trained in scikit-learn package
            
                Example
                
                >>> from sklearn.ensemble import RandomForestRegressor
                >>> rlr = RandomForestRegressor(n_estimators=150, n_jobs=5) #create sklearn model
                >>> rlr.fit(xi_train, y_train) #training model
            
        base_test
            Must be a numpy matrix containing only the fingerprints of all molecules in the test database.
            
                >>> Example of data format
                array([[0., 0., 0., ..., 0., 1., 0.],
                       [0., 0., 0., ..., 0., 0., 0.],
                       [0., 0., 1., ..., 0., 0., 0.],
                       ...,
                       [1., 0., 0., ..., 0., 0., 1.],
                       [0., 1., 1., ..., 1., 0., 0.],
                       [0., 0., 1., ..., 0., 1., 1.]])
        base_train
            Must be a numpy matrix containing only the fingerprints of all molecules in the training database.
            
                >>> Example of data format
                array([[0., 0., 0., ..., 0., 1., 0.],
                       [0., 0., 0., ..., 0., 0., 0.],
                       [0., 0., 1., ..., 0., 0., 0.],
                       ...,
                       [1., 0., 0., ..., 0., 0., 1.],
                       [0., 1., 1., ..., 1., 0., 0.],
                       [0., 0., 1., ..., 0., 1., 1.]])
        y_true
            Must be a numpy array containing only class values of all molecules in the test database
        
        threshold_reference
            The calculation will be carried out in relation to a standard of similarity
            of reference: Maximum value, median, standard deviation or average of
            similarity of the test sample in relation to the training set.
            Set threshold_reference = "max" for maximum, threshold_reference="average" 
            for average value, threshold_reference="str"for standard deviation
            and threshold_reference="median"for median value. ('max', 'average', 'std', 'median')
        
        threshold_step
            Similarity scan range. The minimum similarity is 0 and maximum 1
        
        similarity_metric
            Desired metric to perform the similarity calculation. The options are:
            'tanimoto', 'tversky', 'geometric', 'arithmetic', 'euclidian', 'manhattan'.
            The default metric is tanimoto index.
            The alpha and beta parameters are only for 'tversky' metric.
        
        metric_avaliation
            Desired metric to perform the error, correlation or accuracy calculation. The options are:
            'rmse', 'mse', 'mae', 'mcc', 'acc' and 'auc'.
            
            Use 'rmse', 'mse or 'mae' for regression problems and 'mcc', 'acc' or 'auc' for binary classification problems.
            
            mcc -> matthews correlation coefficient; acc -> accuracy; auc -> area under the curve.
            
            
        """
        
        #reference parameters
        if threshold_reference.lower() == 'max':
            thref = 'Max'
        elif threshold_reference.lower() == 'average':
            thref = 'Mean'
        elif threshold_reference.lower() == 'std':
            thref = 'Std'
        elif threshold_reference.lower() == 'median':
            thref = 'Median'
        else:
            thref = 'Max'
        
        #Get analysis table
        table_anal = ApplicabilityDomain.analyze_similarity(self, base_test=base_test, base_train=base_train,
                                                            similarity_metric=similarity_metric,
                                                            alpha=alpha, beta=beta)
        table_anal.index = np.arange(0, len(table_anal), 1)
        
        results = {}
        total_thresholds = np.arange(threshold_step[0], threshold_step[1], threshold_step[2])
        
        if self.__verbose:
            
            for thresholds in tqdm(total_thresholds):
                samples_GT_threshold = table_anal.loc[table_anal[thref] >= thresholds] #get just samples > threshold
                if len(samples_GT_threshold) == 0:
                    print('\nStopping with Threshold {}. All similarities are less than or equal {} '.format(thresholds, thresholds))
                    break
                samples_LT_threshold = table_anal.loc[table_anal[thref] < thresholds] #get just samples < threshold
                new_xitest = base_test[samples_GT_threshold.index, :] #get samples > threshold in complete base_test
                if isTensorflow:
                    new_ypred = model.predict(new_xitest) #precit y_pred
                    new_ypred[new_ypred <= 0.5] = 0
                    new_ypred[new_ypred > 0.5] = 1
                    new_ypred = new_ypred.astype(int)
                else:
                    new_ypred = model.predict(new_xitest) #precit y_pred
                new_ytrue = y_true[samples_GT_threshold.index] #get y_true (same index of xi_test) (y_true must be a array 1D in this case)
                
                
                #calc of ERROR METRICS (EX: RMSE) or correlation methods
                if metric_avaliation == 'rmse':
                    error_ = self.__smetrics.roots_mean_square_error(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'mse':
                    error_ = self.__smetrics.mean_square_error(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'mae':
                    error_ = self.__smetrics.mean_absolute_error(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'mcc':
                    error_ = matthews_corrcoef(y_true=new_ytrue.ravel(), y_pred=new_ypred.ravel())
                elif metric_avaliation == 'acc':
                    error_ = accuracy_score(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'auc':
                    if isTensorflow:
                        new_yproba = model.predict(new_xitest)                        
                        fp, tp, _ = roc_curve(new_ytrue, new_yproba)
                        error_ = auc(fp, tp)
                    else:
                        new_yproba = model.predict_proba(new_xitest)                        
                        fp, tp, _ = roc_curve(new_ytrue, new_yproba[:, 1])
                        error_ = auc(fp, tp)
                    
                results['Threshold {}'.format(thresholds.round(5))] = [[error_],np.array(samples_LT_threshold.index)]
                
            return results
        
        else:            
            for thresholds in total_thresholds:
                samples_GT_threshold = table_anal.loc[table_anal[thref] >= thresholds] #get just samples > threshold
                if len(samples_GT_threshold) == 0:
                    print('\nStopping with Threshold {}. All similarities are less than or equal {} '.format(thresholds, thresholds))
                    break
                samples_LT_threshold = table_anal.loc[table_anal[thref] < thresholds] #get just samples < threshold
                new_xitest = base_test[samples_GT_threshold.index, :] #get samples > threshold in complete base_test
                if isTensorflow:
                    new_ypred = model.predict(new_xitest) #precit y_pred
                    new_ypred[new_ypred <= 0.5] = 0
                    new_ypred[new_ypred > 0.5] = 1
                    new_ypred = new_ypred.astype(int)
                else:
                    new_ypred = model.predict(new_xitest) #precit y_pred
                new_ytrue = y_true[samples_GT_threshold.index] #get y_true (same index of xi_test) (y_true must be a array 1D in this case)
                
                #calc of ERROR METRICS (EX: RMSE) or correlation methods
                if metric_avaliation == 'rmse':
                    error_ = self.__smetrics.roots_mean_square_error(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'mse':
                    error_ = self.__smetrics.mean_square_error(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'mae':
                    error_ = self.__smetrics.mean_absolute_error(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'mcc':
                    error_ = matthews_corrcoef(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'acc':
                    error_ = accuracy_score(y_true=new_ytrue, y_pred=new_ypred)
                elif metric_avaliation == 'auc':
                    if isTensorflow:
                        new_yproba = model.predict(new_xitest)
                        fp, tp, _ = roc_curve(new_ytrue, new_yproba)
                        error_ = auc(fp, tp)
                    else:
                        new_yproba = model.predict_proba(new_xitest)
                        fp, tp, _ = roc_curve(new_ytrue, new_yproba[:, 1])
                        error_ = auc(fp, tp)
                    
                results['Threshold {}'.format(thresholds.round(5))] = [[error_],np.array(samples_LT_threshold.index)]
                
            return results
        

# Teste Zone 1
"""if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from pyADA import Similarity, ApplicabilityDomain
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                                 roc_curve, auc, hamming_loss, classification_report,
                                 roc_auc_score)
    
    from tensorflow.keras.models import load_model
    import os

    def filter_data(df_new, colnames, filter_threshold = 2000, seed=0, balance_0s = True):
        for i in colnames:
            if i == colnames[0]:
                old = df_new.loc[df_new[i] == 1]
            else:
                ocorr_new_in_old = len(old.loc[old[i] == 1])

                if df_new[i].isin([1]).sum() <= filter_threshold:
                    if ocorr_new_in_old == 0:
                        new = df_new.loc[df_new[i] == 1]
                        new = pd.concat([old, new], axis=0)
                        old = new
                    else:
                        new = df_new.loc[df_new[i] == 1]
                        new.drop(old.loc[old[i] == 1].index, inplace=True, axis = 0)
                        new = pd.concat([old, new])
                        old = new
                else:            
                    if ocorr_new_in_old >= filter_threshold:
                        continue
                    else:
                        new = df_new.loc[df_new[i] == 1]
                        new.drop(old.loc[old[i] == 1].index, axis=0, inplace=True)
                        new = new.sample(filter_threshold - ocorr_new_in_old, random_state=seed)
                        new = pd.concat([old, new])
                        old = new
                        
        #ainda pegamos só 3 colunas, depois temos que deixar esse código a baixo geral (talvez usando pd.query)
        if balance_0s:
            get_all0 = df_new.loc[(df_new[colnames[0]] == 0) & (df_new[colnames[1]] == 0) & (df_new[colnames[2]] == 0)]
            get_filt0 = get_all0.sample(filter_threshold//3, random_state=seed)
            old = pd.concat([old, get_filt0], axis=0)
            
            
        return old
    
    #Load data
    path = '/data/banco_de_dados/AIRNmr/full_204FG.csv'
    df = pd.read_csv(path)
    togetY = [ 0, 1, 2, 3, 4, 8, 15, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 37, 38, 40, 44, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 60, 62, 63, 64, 65, 67, 68, 69, 70, 74, 75, 77, 78, 79, 81, 82, 83, 89, 98, 105, 112, 114, 124, 126, 132, 149, 160, 163, 165, 172, 198, 199, 200, 201]
    y = df.iloc[:, 2403:].iloc[:, togetY]
    df_new = pd.concat([df.iloc[:, :2403], y], axis=1)
    #Load Models
    path_models = '/dados/GoogleDrive/rotinas_python/AIRNmr/app/prenumars/models/'
    models = {}
    for i in os.listdir(path_models):
        models[i] = path_models+i
    model = load_model(models['model_FG27.h5'])
    #Get correct fingerprints
    fgs_names = ['FG-27', 'FG-4', 'FG-46']
    df_filter = filter_data(df_new=df_new, colnames=fgs_names, filter_threshold=4385, balance_0s=False)
    df_filter = df_filter.sample(len(df_filter), random_state=0)
    #Prepare data
    xi = df_filter.iloc[:, 3:2403].astype(int).values
    y = df_filter.loc[:, ['FG-27']].values
    new_xi = xi
    column2add = np.zeros((len(xi),1)) #criando a coluna que vai ser adicionada à matriz
    new_xi = np.append(new_xi, column2add, axis=1) #adicionando a coluna nova na matriz
    xi49_49 = new_xi.reshape(len(new_xi), 49, 49)
    xi49_49 = new_xi.reshape(len(new_xi), 49, 49, 1).astype(int)
    #Train and test
    xi_train, xi_test, y_train, y_test = train_test_split(xi49_49, y, test_size=0.2, random_state=487)
    #AD
    AD = ApplicabilityDomain(verbose=True)
    get = AD.fit(model=model, isTensorflow=True,base_test=xi_test[:100], base_train=xi_train, y_true=y_test,
            threshold_reference = 'max', threshold_step = (0, 1, 0.05),
            similarity_metric='tanimoto', alpha = 1, beta = 1, metric_avaliation='mcc')"""

    

    



# Teste Zone 2
"""if __name__ == '__main__':
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelEncoder

    import pandas as pd
    ## Regression
    #path_train = '/dados/ShareGDrive/MachineLearningRateConstant/dados/kOH_RC_FP.csv'
    #df = pd.read_csv(path_train)
    #df_train = df.iloc[:500, 2:]
    #df_test = df.iloc[500:, 2:]
    #xi_train = df_train.iloc[:, 1:].values
    #y_train = df_train.iloc[:, 0].values
    #xi_test = df_test.iloc[:, 1:].values
    #y_test = df_test.iloc[:, 0].values
    #rlr = RandomForestRegressor(n_estimators=150, n_jobs=5)
    #rlr.fit(xi_train, y_train)
    #rlr.predict([xi_test[0]])
    #y_pred = rlr.predict(xi_test)
    
    ## Classification
    le = LabelEncoder()
    path = '/data/banco_de_dados/Machine_Learning_UCI/qsar_toxicitydrugs/qsar_oral_toxicity.csv'
    df = pd.read_csv(path, sep=';', header=None)    
    xi = df.iloc[:, 0:1024].values
    y = df.iloc[:, 1024].values
    y = le.fit_transform(y)

    xi_train, xi_test, y_train, y_test = train_test_split(xi, y, test_size=0.05, random_state=0)    
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=10, random_state=0)
    rfc.fit(xi_train, y_train)
    

    ad3 = ApplicabilityDomain(verbose=True)
    get = ad3.fit(model=rfc,base_test=xi_test, base_train=xi_train, y_true=y_test,
                  threshold_step=(0, 1, 0.1), threshold_reference='max',
                  metric_avaliation='auc', similarity_metric='tanimoto', alpha=1, beta=1)"""



    























