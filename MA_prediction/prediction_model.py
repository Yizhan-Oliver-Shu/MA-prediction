import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

import sklearn
from sklearn import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

scorer_dict = {"average_precision": average_precision_score, "roc_auc": roc_auc_score}

def check_scoring_type(scoring):
    scoring_is_str, scoring_is_iter =  isinstance(scoring, str), isinstance(scoring, (list, tuple, set))
    if not scoring_is_str and not scoring_is_iter:
        raise Exception("Input parameter scoring must be either a string, or an iterable of strings!")  
    return scoring_is_str, scoring_is_iter
    
    
def score_prediction(y, y_pred, scoring = "roc_auc"):
    """
    calculating the score of a prediction, where
    - y may contain more index than y_pred, so you don't have to index y before inputting,
    - pending deals are neglected in calculation.
    - scoring can be an iterable of strings, where the output is a dict of scores.
    
    y is assumed to be {0: "C", 1: "W", 99: "P"}.
    
    Parameters:
    ---------------------------------
    y: Series
        the true label. can contain more indices that those of `y_pred`. can contain 99, that is unknown results, neglected in scoring
    y_pred: Series
        prediction.
    scoring: str or iterable of strings.
        now supports {"roc_auc", "average_precision"}
        
    Returns:
    ---------------------------------
    float or dict of scores.
    """
    scoring_is_str, scoring_is_iter = check_scoring_type(scoring)
    # if not scoring_is_str and not scoring_is_iter:
    #     raise Exception("Input parameter scoring must be either a string, or an iterable of strings!")
        
    if y_pred.empty:
        return np.nan if scoring_is_str else {scorer: np.nan for scorer in scoring}
    
    # extract sub series
    y_sub = y.loc[y_pred.index]
    # exclude pending deals when calculating scores
    idx = y_sub.ne(99)
    y_true, y_prediction = y_sub[idx], y_pred[idx]
    if scoring_is_str:
        return scorer_dict[scoring](y_true, y_prediction)
    else: # return a dict
        return {scorer: scorer_dict[scorer](y_true, y_prediction) for scorer in scoring}
            

def generate_train_test_yr_split(start_yr = 1990,
                                 end_yr = 2022,
                                 test_start_yr = None,
                                 num_train_yrs = 12,
                                 num_test_yrs = 1,
                                 window="expanding"
                                ):
    """
    Generate the train/validation time series split. 
    Default: 1990-2022, expanding window, start validation at 2002, each validation set is one year. Then this would be:
    - fold 0: train on (1990-2001), validate on 2002,
    - fold 1: train on (1990-2002), validate on 2003,
    ...
    
    Parameters:
    ------------------------------
    start_yr: int, default 1990.
        start year of the whole model selection/validation process.
    end_yr: int, default 2022.
        end year of the whole model selection/validation process.
    test_start_yr: int, default None.
        the start year of testing. The real test start year can be inferred from either this parameter and <num_train_yrs>
    num_train_yrs: int, default 12,
        number of years of data in each training set for rolling window, and the number of years of data in the first/smallest training set for expanding window.
    num_test_yrs: int, default 1,
        number of years of data in each validation set.
    window: string, {"rolling", "expanding"}
        
    Returns:
    ------------------------------
    A list of ((train_start, train_end), (test_start, test_end)
    """
    res = []
    if window not in ["rolling", "expanding"]:
        raise Exception("Unsupported input for rolling/expanding window!")
    rolling_window = (window == "rolling")
    # infer the start of test
    if num_train_yrs:
        test_start_yr2 = num_train_yrs + start_yr
    else:
        test_start_yr2 = -1
    if not test_start_yr:
        test_start_yr = -1
    test_start_yr = max(test_start_yr, test_start_yr2)
    if test_start_yr <= start_yr or test_start_yr > end_yr:
        raise Exception("Invalid input regards of test start year!")
        
    # return test_start_yr
    for test_start in range(test_start_yr, end_yr + 1, num_test_yrs):  # iterate over folds
        test_end = min(end_yr, test_start + num_test_yrs - 1)
        train_end = test_start - 1
        train_start = test_start - num_train_yrs if rolling_window else start_yr
        res.append(((train_start, train_end), (test_start, test_end)))
    return res

def generate_train_test_idx_split(y, train_test_yr_split, da, dr):
    """
    generate the indices for each train/test split, as a list of (train_index, test_index) from the train/test year split. 
    y can contain more indices that those of `da` and `dr`, meaning that you can input the whole `da` and `dr` series.
    
    Split criterion:
    - for training set, da_yr >= train_start and dr_yr <= train_end, and no pending deals, i.e. we must know the result of completion/termination before the training end year. 
    - for test set, test_start <= dr_yr <= test_end.
    """
    da_sub, dr_sub = da.loc[y.index], dr.loc[y.index]
    # transform date to year
    f = (lambda x: x.year)
    da_yr, dr_yr = da_sub.map(f), dr_sub.map(f)
    # res
    train_test_idx_split = []
    for (train_start, train_end), (test_start, test_end) in train_test_yr_split:
        train_index, test_index = y.index[da_yr.ge(train_start)&dr_yr.le(train_end)&y.ne(99)], y.index[da_yr.between(test_start, test_end)]
        train_test_idx_split.append((train_index, test_index))
    return train_test_idx_split

def fit_predict_CV_split(estimator, param_grid, train_test_idx_split, X, y, return_train_pred=True):
    """
    on each fold of cv split, fit the model on training data with all the params in the grid, and predict on the test (or train) set, 
    save the prediction result as a Series in a 2d list, with the axis 0 being param, and axis 1 being folds.
    """
    if type(param_grid) != sklearn.model_selection.ParameterGrid:
        PG = ParameterGrid(param_grid)
    else:
        PG = param_grid
        
    N_params, N_folds = len(PG), len(train_test_idx_split)
    # initiate a 2d array
    y_pred_test_list = [[np.nan for _ in range(N_folds)] for _ in range(N_params)]
    if return_train_pred:
        y_pred_train_list = [[np.nan for _ in range(N_folds)] for _ in range(N_params)]
    # iter over folds
    for n_fold, (train_index, test_index) in enumerate(tqdm(train_test_idx_split)):
        X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
        # iter over all param combinations
        for n_param, param in enumerate(PG):
            # clone an estimator, and set the params
            new_model = clone(estimator).set_params(**param)
            # fit on training set
            new_model = new_model.fit(X_train, y_train)
            # print(f"fold {n_fold}, param {param}")
            # predict on test
            y_pred_test = new_model.predict_proba(X_test)[:, 1]
            y_pred_test_list[n_param][n_fold] = pd.Series(y_pred_test, index=test_index)
            if return_train_pred:
                y_pred_train = new_model.predict_proba(X_train)[:, 1]
                y_pred_train_list[n_param][n_fold] = pd.Series(y_pred_train, index=train_index)
    return (y_pred_test_list, y_pred_train_list) if return_train_pred else y_pred_test_list


def combine_val_pred(y_pred_test_list, n_vals=3):
    """
    combine the prediction results on several folds, (to later compute one validation score on a larger validation set)
    """
    N_params, N_folds = len(y_pred_test_list), len(y_pred_test_list[0])
    combined_pred_list = [[np.nan for _ in range(N_folds)] for _ in range(N_params)]  
    for n_param in range(N_params):
        for n_fold in range(N_folds):
            if n_fold < n_vals:
                combined_pred_list[n_param][n_fold] = pd.Series(dtype=float)
            else:
                combined_pred_list[n_param][n_fold] = pd.concat(y_pred_test_list[n_param][(n_fold-n_vals):n_fold])
    return combined_pred_list

def score_y_pred_list(y, y_pred_list, scoring="average_precision"):
    """
    score a list of predicitons
    """
    scoring_is_str, scoring_is_iter = check_scoring_type(scoring)
    if scoring_is_str:
        N_params, N_folds = len(y_pred_list), len(y_pred_list[0])
        scores = np.zeros((N_params, N_folds))
        for n_param, n_fold in product(range(N_params), range(N_folds)):
            scores[n_param, n_fold] = score_prediction(y, y_pred_list[n_param][n_fold], scoring)
        return scores
    else: # return a dict
        return {scorer: score_y_pred_list(y, y_pred_list, scorer) for scorer in scoring}
    
class GridSearch_TS_CV():
    def __init__(self,
                 estimator, 
                 param_grid, 
                 scoring=None, 
                 criterion=None, 
                 return_train_score=True,
                 start_yr=1990,
                 end_yr=2022,
                 test_start_yr=None,
                 num_train_yrs=12,
                 num_test_yrs=1,
                 num_val_folds=3,
                 window='expanding'):
        self.estimator = estimator
        self.param_grid = ParameterGrid(param_grid)
        self.scoring = scoring
        self.criterion = criterion
        self.return_train_score = return_train_score
        self.yr_split = generate_train_test_yr_split(start_yr, end_yr, test_start_yr, num_train_yrs, num_test_yrs, window)
        self.num_val_folds = num_val_folds
    
    def fit(self, X, y, da, dr):
        # get attributes
        yr_split, estimator, param_grid, return_train_score, num_val_folds, scoring = self.yr_split, self.estimator, self.param_grid, self.return_train_score, self.num_val_folds, self.scoring
        criterion = self.criterion
        # generate idx split
        idx_split = generate_train_test_idx_split(y, yr_split, da, dr)
        # prediction result
        res = fit_predict_CV_split(estimator, param_grid, idx_split, X, y, return_train_score)
        if return_train_score:
            y_pred_test_list, y_pred_train_list = res
        else:
            y_pred_test_list = res
        # combine val preds
        y_pred_val_list = combine_val_pred(y_pred_test_list, num_val_folds)
        # compute val scores
        val_scores = score_y_pred_list(y, y_pred_val_list, scoring)
        val_scores_final = val_scores if not criterion else val_scores[criterion]
        best_index_ = val_scores_final.argmax(axis=0)
        best_params_ = [param_grid[idx] for idx in best_index_]
        # best scores
        best_val_scores_ = extract_best_scores(val_scores, best_index_)
        
        # final prediction
        y_pred_test = []
        for n_fold in range(num_val_folds, len(yr_split)):
            y_pred_test.append(y_pred_test_list[best_index_[n_fold]][n_fold])
        y_pred_test = pd.concat(y_pred_test)
        test_scores = score_prediction(y, y_pred_test, scoring)
        
        # train scores
        if return_train_score:
            train_scores = score_y_pred_list(y, y_pred_train_list, scoring)
            best_train_scores_ = extract_best_scores(train_scores, best_index_)        

        
        
        # save results
        self.idx_split = idx_split
        self.y_pred_test_list = y_pred_test_list
        self.val_scores = val_scores
        self.best_val_scores_ = best_val_scores_
        self.y_pred_val_list = y_pred_val_list
        self.best_index_ = best_index_
        self.y_pred_test = y_pred_test
        self.test_scores = test_scores
        self.best_params_ = best_params_
        if return_train_score:
            self.y_pred_train_list = y_pred_train_list
            self.train_scores = train_scores        
            self.best_train_scores_ = best_train_scores_
        
        return self
    
def extract_best_scores(scores, best_index_):
    if isinstance(scores, np.ndarray):
        return scores[best_index_, range(scores.shape[1])]
    else:
        return {scorer:extract_best_scores(scores[scorer], best_index_) for scorer in scores}