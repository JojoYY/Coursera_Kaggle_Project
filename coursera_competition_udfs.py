import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import pickle
import lightgbm as lgb
from itertools import product
from functools import reduce
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


PATH_TO_DATA = '/home/johanna/Data/Coursera_Kaggle_Project'


def downcast_dtypes(df):
    '''
        Changes column types in the dataframe:

                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df


### UDFs for loading raw data ###

def clean_sales_data(path_to_data=PATH_TO_DATA, level=None):
    df_sales = pd.read_csv(path_to_data + '/sales_train_v2.csv')

    # drop the 6 duplicated rows
    df_sales = df_sales.drop_duplicates()

    # convert item count data to integer
    df_sales['item_cnt_day'] = df_sales['item_cnt_day'].astype(int)
    
    # correct data type
    df_sales['date'] = pd.to_datetime(df_sales['date'], format='%d.%m.%Y')
    return df_sales

    
def clean_items_data(path_to_data=PATH_TO_DATA):
    items_data = pd.read_csv(path_to_data + '/items.csv')
    items_data['item_name_proc'] = items_data['item_name'].apply(lambda x: x.lstrip(' \"!*/'))
    return items_data


def clean_items_categ_data(path_to_data=PATH_TO_DATA):
    item_categ_data = pd.read_csv(path_to_data + '/item_categories.csv')

    # Translations from google translate
    item_categ_names_eng = pd.read_table('item_categories_names_translated.txt', header=None)

    item_categ_data['item_category_name_eng'] = item_categ_names_eng
    item_categ_data['item_category_group'] = (item_categ_data['item_category_name_eng']
                                                  .apply(lambda x: x.split(' - ')[0])
                                                      .apply(lambda x: x.lower())
                                             )
    
    return item_categ_data


def clean_shops_data(path_to_data=PATH_TO_DATA):
    shops_data = pd.read_csv(path_to_data + '/shops.csv')

    # Translations from google translate
    shops_data['shop_name_eng'] = pd.read_table('shops_translated.txt', header=None)

    # standardize shop names to extract further info
    shops_data['shop_name_eng'] = shops_data['shop_name_eng'].apply(lambda x: x.lower())
    shops_data['shop_city'] = (shops_data['shop_name_eng'].apply(lambda x: x.split(' ')[0]))
    return shops_data


def merge_items_data(path_to_data=PATH_TO_DATA):
    items = clean_items_data(path_to_data)
    categories = clean_items_categ_data(path_to_data)
    
    item_cols = ['item_id', 'item_category_id']
    cat_cols = ['item_category_id', 'item_category_name_eng', 'item_category_group']
    
    item_all = pd.merge(items[item_cols], categories[cat_cols], on='item_category_id', how='left')
    item_all.drop_duplicates(inplace=True)
    return item_all
    

def make_monthly_features_from_sales(sales):
    df_sales = sales.copy()
    df_sales['year'] = df_sales['date'].dt.year
    df_sales['month'] = df_sales['date'].dt.month

    month_to_quarters = {1: 'Q1', 2: 'Q1', 3: 'Q1',
                         4: 'Q2', 5: 'Q2', 6: 'Q2',
                         7: 'Q3', 8: 'Q3', 9: 'Q3',
                         10: 'Q4', 11: 'Q4', 12: 'Q4'}
    df_sales['quarters'] = df_sales['month'].map(month_to_quarters)
    df_sales['december'] = df_sales['date_block_num'].apply(lambda x: x in [11, 23])
    return df_sales


def make_daily_features_from_sales(sales):
    df_sales = sales.copy()
    df_sales['day_of_month'] = df_sales['date'].dt.day
    df_sales['weekday_name'] = df_sales['date'].dt.weekday_name
    df_sales['day_of_week'] = df_sales['date'].dt.weekday
    return df_sales


def add_item_category_features(sales, path_to_data=PATH_TO_DATA):
    # data sets to merge
    df = sales.copy()
    items = clean_items_data(path_to_data)
    categories = clean_items_categ_data(path_to_data)
    
    item_cols = ['item_id', 'item_category_id']
    item_categ_cols = ['item_category_id', 'item_category_group']
    
    items_all = pd.merge(items[item_cols], categories[item_categ_cols], on='item_category_id', how='left')
    items_all.drop_duplicates(inplace=True)
    return pd.merge(df, items_all, on='item_id', how='left')
    
    
def add_shop_features(sales, path_to_data=PATH_TO_DATA):
    # data sets to merge
    df = sales.copy()
    shops_data = clean_shops_data(path_to_data)
    #shops_data['shop_id'] = shops_data['shop_id'].astype('category')
    
    shop_cols = ['shop_id', 'shop_city']
    return df.merge(shops_data[shop_cols], on='shop_id', how='left')
    

def merge_all_train_data(path_to_data=PATH_TO_DATA):
    sales_train_data = clean_sales_data(path_to_data)
    items_data = clean_items_data(path_to_data)
    item_categ_data = clean_items_categ_data(path_to_data)
    shops_data = clean_shops_data(path_to_data)

    item_cols = ['item_id', 'item_name_proc', 'item_category_id']
    # we drop original names since translations are clear
    item_categ_cols = ['item_category_id', 'item_category_name_eng', 'item_category_group']

    items_all = pd.merge(items_data[item_cols], item_categ_data[item_categ_cols],
                         on='item_category_id')

    item_all_cols = np.unique(item_cols + item_categ_cols)

    # we keep original shop name bc we may be able to extract more informaton
    # (do TU and TPU both stand for shopping center?)
    shop_cols = ['shop_id', 'shop_name', 'shop_name_eng', 'shop_city']
    all_train = (sales_train_data
                 .merge(shops_data[shop_cols], on='shop_id')
                 .merge(items_all, on='item_id'))

    # rearrange columns so same information is grouped together
    date_cols = ['date', 'date_block_num', 'year', 'month', 'day_of_month', 'month_year']
    sale_cols = ['item_price', 'item_cnt_day']

    all_train = all_train[date_cols + shop_cols + item_all_cols.tolist() + sale_cols]
    return all_train


def load_test_data(path_to_data=PATH_TO_DATA):
    test_data = pd.read_csv(path_to_data + '/test.csv')
    return test_data


def load_submission_file(path_to_data=PATH_TO_DATA):
    sample_submission_data = pd.read_csv(path_to_data + '/sample_submission.csv')
    return sample_submission_data

### UDFs for aggregating sales data by month ####
def create_shop_item_grid_by_month(sales):
    """Creates a dataframe with all shop-item indices available each month.

    Takes as input a dataframe with the structure of sales_train.csv, possiobly limited
    to certain month, shops and items. Returns a new dataframe containing for each month in
    the input dataframe, all the combinations of shops and items that appear in that month
    in the input dataframe.

    It does not contain sales data, just the indices for month, shop and item as a grid.
    Use this for merging with sales data aggregated at the month level.
    """
    grid = []
    for block_num in sales['date_block_num'].unique():
        cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
        cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]]))))

    # Turn the grid into a dataframe
    return pd.DataFrame(np.vstack(grid),
                        columns=['shop_id', 'item_id', 'date_block_num'])

  
def create_monthly_sales_grid(sales):
    """Aggregate monthly sales and merge to (shop, item) grid."""

    shop_item_month = ['shop_id', 'item_id', 'date_block_num']
    shop_month = ['shop_id', 'date_block_num']
    item_month = ['item_id', 'date_block_num']

    monthly_grid = create_shop_item_grid_by_month(sales[shop_item_month])
    
    target_shop_item = (sales.groupby(shop_item_month, as_index=False)
                            .agg({'item_cnt_day': 'sum'})
                                .rename(columns={'item_cnt_day': 'target'})
                       )

    all_data = pd.merge(monthly_grid, target_shop_item, how='left', on=shop_item_month).fillna(0)
    print('column added:', 'target')

    target_by_shop = (sales.groupby(shop_month, as_index=False)
                          .agg({'item_cnt_day': 'sum'})
                              .rename(columns={'item_cnt_day': 'target_shop'})
                     )
    all_data = pd.merge(all_data, target_by_shop, how='left', on=shop_month).fillna(0)
    print('column added:', 'target_shop')
        
    target_by_item = (sales.groupby(item_month, as_index=False)
                          .agg({'item_cnt_day': 'sum'})
                              .rename(columns={'item_cnt_day': 'target_item'})
                     )
    all_data = pd.merge(all_data, target_by_item, how='left', on=item_month).fillna(0)
    print('column added:', 'target_item')
    
    # convert item count data to integer
    all_data['target'] = all_data['target'].astype(int)
    all_data = downcast_dtypes(all_data)    
    del monthly_grid, target_shop_item, target_by_shop, target_by_item
    gc.collect()

    return all_data


def create_lagged_sales(all_data, shift_range, cols_to_shift):
    df = all_data.copy()
    index_cols = ['shop_id', 'item_id', 'date_block_num']
    
    for month_shift in shift_range:
        print('Calculating shift by %d periods' % month_shift)    
        train_shift = df[index_cols + cols_to_shift].copy()
        # shift index to create block of lagged series
        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
        # reset column names to reflect lagging
        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_shift else x
        train_shift = train_shift.rename(columns=foo)
        # merge lagged columns to input dataframe
        df = pd.merge(df, train_shift, on=index_cols, how='left')
        
    # List of all lagged features
    lag_cols = [col for col in df.columns if col[-1] in [str(item) for item in shift_range]]
    # for (shop, item) pairs not available at lagged month, fill with 0
    df[lag_cols] = df[lag_cols].fillna(0).astype(int)   
    
    # this fills unavailable lags with artificial 0, delete rows for training
    print('Deleting date_block_num < %d from data' % np.max(shift_range))
    df = df.loc[df['date_block_num']>= np.max(shift_range)]
    
    # it also fills unvailable target values 
    del train_shift
    gc.collect()
    return df


def create_all_monthly_sales_grid_with_lags(lags=None,
                                            cols_to_lag=['target', 'target_item', 'target_shop'],
                                            path_to_data=PATH_TO_DATA):
    sales = clean_sales_data(path_to_data)  # labelled training data
    test_data = load_test_data(path_to_data)  # test data without labels
    
    # aggregated monthly sales for (shop, item) grids by month
    sales_grid = create_monthly_sales_grid(sales)
    
    # extract monthly (shop, item) grid from test data
    test_grid = test_data[['shop_id', 'item_id']].copy()
    test_grid['date_block_num'] = 34
    
    # concatenate train and test grid for feature generation
    all_grid = pd.concat([sales_grid, test_grid], ignore_index=True, sort=True)

    if lags:
        all_sales = create_lagged_sales(all_grid, shift_range=lags, cols_to_shift=cols_to_lag)
    del sales, test_data, all_grid
    gc.collect()
    
    # delete unavailable information about label at the time
    all_sales.drop(columns=['target_item', 'target_shop'], inplace=True)

    #all_sales['shop_id'] = all_sales['shop_id'].astype('category')
    return all_sales
    

def add_seasonality_features(sales_data):
    sales = sales_data.copy()

   # hard-coded mapping from date_block_num to month of the year (derived from original sales train data)

    months = np.tile(np.arange(1, 13), 3)[:-1]
    date_blocks = np.arange(35)
    date_block_to_month = pd.Series(months, index=date_blocks)

    sales['month'] = sales['date_block_num'].map(date_block_to_month)
    sales['december'] = (sales['month']==12)

    month_to_quarters = {1: 'Q1', 2: 'Q1', 3: 'Q1',
                         4: 'Q2', 5: 'Q2', 6: 'Q2',
                         7: 'Q3', 8: 'Q3', 9: 'Q3',
                         10: 'Q4', 11: 'Q4', 12: 'Q4'}

    sales['quarters'] = sales['month'].map(month_to_quarters).astype('category')

    sales['Q1'] = (sales['quarters']=='Q1')
    sales['Q4'] = (sales['quarters']=='Q4')
    
    # remove month and quarter information to keep only the major ones
    sales.drop(columns=['month', 'quarters'], inplace=True)
    return sales


def add_sales_features(sales):
    # add features to single out no sales
    df = sales.copy()
    df['target_shop_zero'] = (df['target_shop_lag_1']==0)

    df['target_item_zero'] = (df['target_item_lag_1']==0)

    df['target_shop_recent_zero'] = ((df['target_shop_lag_1']==0) &
                                            (df['target_shop_lag_2']==0) & 
                                            (df['target_shop_lag_3']==0))

    df['target_item_recent_zero'] = ((df['target_item_lag_1']==0) &
                                            (df['target_item_lag_2']==0) & 
                                            (df['target_item_lag_3']==0))
    return df


def load_all_monthly_sales(lag_months=[1, 2, 3, 6, 12], seasonality=True, items=True, shops=True):
    filename = 'monthly_sales_grid_lag_' + '_'.join(str(x) for x in lag_months)
    all_sales = create_all_monthly_sales_grid_with_lags(lags=lag_months)
    all_sales = add_sales_features(all_sales)
    
    if seasonality:
        all_sales = add_seasonality_features(all_sales)
        filename += '_seasonality'
    
    if items:
        all_sales = add_item_category_features(all_sales)
        filename += '_items'
    
    if shops:
        all_sales = add_shop_features(all_sales)
        filename += '_shops'
    pkl_name = './' + filename + '.pkl'
    print('generate and pickling file', pkl_name)
    all_sales.to_pickle(pkl_name)
    return all_sales


def log_transform(s):
    # yields a distribution starting at zero
    s_min = s.min()
    s_trans = np.log(s-s_min+1)
    return s_trans, s_min


def inv_log_transform(s_trans, s_min):
    return np.exp(s_trans+s_min-1)


def label_encode_categ_cols(df, categ_cols):
    proc = df.copy()
    for col in categ_cols:
        filename = 'enc_' + col + '.pickle'
        print('label encoding %s and saving to %s' % (col, filename))
        # training the encoding
        enc = LabelEncoder().fit(proc[col])
        # save the encoding for future reversing
        with open(filename, 'wb') as out_file:
            pickle.dump(enc, out_file, pickle.HIGHEST_PROTOCOL)
        # apply encoding
        proc[col] = enc.transform(proc[col])
    return proc


def inverse_label_encode_categ_cols(df, categ_cols):
    # this only works in combination with the label_encode_categ_cols() udf!
    proc = df.copy()
    for col in categ_cols:
        filename = 'enc_' + col + '.pickle'
        print('inverse label encoding %s from file %s'% (col, filename))        
        with open(filename, 'rb') as in_file:
            enc_inv = pickle.load(in_file)
            proc[col] = enc_inv.inverse_transform(proc[col])
    return proc
    
    
def train_test_split_by_month(df_all, test_start=33, label='target'):
    # Split labelled training data into train and validation sets
    first_block = df_all.loc[df_all[label].notnull(), 'date_block_num'].min()
    if test_start<34:
        last_block = df_all.loc[df_all[label].notnull(), 'date_block_num'].max()
    else:
        last_block=34

    train_blocks = (df_all['date_block_num']>=first_block) & (df_all['date_block_num']<=test_start-1)
    test_blocks = (df_all['date_block_num']>=test_start) & (df_all['date_block_num']<=last_block)

    df_train = df_all.loc[train_blocks]
    df_test = df_all.loc[test_blocks]

    print('Train data: date_block_num', df_train['date_block_num'].unique())
    print('Test data: date_block_num', df_test['date_block_num'].unique())
    
    X_train, y_train = df_train.drop(columns=[label]), df_train[label]
    X_test, y_test = df_test.drop(columns=[label]), df_test[label]

    print('Number of observations in train:', X_train.shape[0])
    print('Number of observations in test:', X_test.shape[0])
    print('Number of attributes:', X_test.shape[1])
    
    del df_train, df_test
    gc.collect()
    return X_train, X_test, y_train, y_test


def fit_eval_model(model, X_train, X_val, y_train, y_val, clip_val=True):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)      
    
    # truncate predictions to [0, 20] as for test data
    if clip_val:
        pred_val = np.clip(pred_val, 0, 20)
    
    rsquared = [r2_score(y_train, pred_train), r2_score(y_val, pred_val)]
    rmse = [np.sqrt(mean_squared_error(y_train, pred_train)), np.sqrt(mean_squared_error(y_val, pred_val))]

    print(model.__class__.__name__)
    df_out = pd.DataFrame({'R-squared': np.round(rsquared, 3), 'RMSE': np.round(rmse, 3)},
                      index=['train', 'val'])
    print(df_out)
    return model


class TrainLgbm(object):

    def __init__(self, lgb_params, X_train, y_train, num_rounds, X_val, y_val,
                 categoricals, stopping_rounds=20, verbose_eval=20):
        self.lgb_params = lgb_params
        self.X_train = X_train
        self.y_train = y_train
        self.num_rounds = num_rounds
        self.X_val = X_val
        self.y_val = y_val
        self.categoricals = categoricals
        self.stopping_rounds = stopping_rounds
        self.evals_result = {}
        self.verbose_eval = verbose_eval

    def train_early_stop(self):
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_val = lgb.Dataset(self.X_val, self.y_val, reference=lgb_train)

        bst = lgb.train(params = self.lgb_params,
                        train_set = lgb_train,
                        num_boost_round = self.num_rounds,
                        valid_sets=[lgb_train, lgb_val],
                        valid_names=['train', 'val'],
                        categorical_feature=self.categoricals,
                        early_stopping_rounds=self.stopping_rounds,
                        evals_result=self.evals_result,
                        verbose_eval=self.verbose_eval,
                        keep_training_booster=True)

        # extract predictions of the best model for further analysis
        self.pred_train = bst._Booster__inner_predict(data_idx=0)
        self.pred_val = bst._Booster__inner_predict(data_idx=1)
        return bst
    
    def train_test_metrics(self, clip=False):
        print('Performance metrics on train and val set')
        if clip:
            print('\tafter clipping all target values to [0, 20] range')
            y_train = np.clip(self.y_train, 0, 20)
            y_val = np.clip(self.y_val, 0, 20)
            pred_train = np.clip(self.pred_train, 0, 20)
            pred_val = np.clip(self.pred_val, 0, 20)
        else:
            print('\tusing given target values')
            y_train = self.y_train
            y_val = self.y_val
            pred_train = self.pred_train
            pred_val = self.pred_val

        rsquared = [r2_score(y_train, pred_train),
                    r2_score(y_val, pred_val)]

        rmse = [np.sqrt(mean_squared_error(y_train, pred_train)),
                np.sqrt(mean_squared_error(y_val, pred_val))]

        df_out = pd.DataFrame({'R-squared': np.round(rsquared, 3),
                               'RMSE': np.round(rmse, 3)},
                              index=['train', 'val'])
        return df_out
    
    
    def error_analysis(self, data='val'):
        print('labels and predictions are clipped to [0,20]')
        if data=='val':
            print('residuals on validation data')
            label = np.clip(self.y_val, 0, 20)
            predicts = np.clip(self.pred_val, 0, 20)
        else:
            print('residuals on training data')
            label = np.clip(self.y_train, 0, 20)
            predicts = np.clip(self.pred_train, 0, 20)

        df = pd.DataFrame({'y_clip' : label,
                            'pred_clip' : predicts})

        df['residual'] = df['y_clip'] - df['pred_clip']
        df.boxplot(column='residual', by='y_clip', figsize=(8,4))
        plt.show()
        
        df['residual_squared'] = df['residual']**2
        
        (100*df.groupby('y_clip')['residual_squared'].sum()/df['residual_squared'].sum()).plot(kind='bar')
        plt.ylabel('residual_squared')
        plt.show()
        
        df['true_zero'] = (df['y_clip']==0)
        df['pred_zero'] = (df['pred_clip']==0)
        print(confusion_matrix(df['true_zero'], df['pred_zero']))
        return df
