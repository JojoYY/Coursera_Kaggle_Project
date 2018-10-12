import numpy as np
import pandas as pd

path_to_data = '/home/johanna/Data/Coursera_Kaggle_Project'


def clean_sales_data():
    sales_train_data = pd.read_csv(path_to_data + '/sales_train_v2.csv')

    # Correct data format
    sales_train_data['date'] = pd.to_datetime(sales_train_data['date'], format='%d.%m.%Y')

    # convert item count data to integer
    sales_train_data['item_cnt_day'] = sales_train_data['item_cnt_day'].astype(int)

    # drop the 6 duplicated rows
    sales_train_data = sales_train_data.drop_duplicates()

    # extracting date information
    sales_train_data['year'] = sales_train_data['date'].dt.year
    sales_train_data['month'] = sales_train_data['date'].dt.month
    sales_train_data['day_of_month'] = sales_train_data['date'].dt.day

    # This takes some time to run
    sales_train_data['month_year'] = sales_train_data['date'].dt.to_period('M')

    sales_train_data['weekday_name'] = sales_train_data['date'].dt.weekday_name
    sales_train_data['day_of_week'] = sales_train_data['date'].dt.weekday
    return sales_train_data


def clean_items_data():
    items_data = pd.read_csv(path_to_data + '/items.csv')
    items_data['item_name_proc'] = items_data['item_name'].apply(lambda x: x.lstrip(' \"!*/'))
    return items_data


def clean_items_categ_data():
    item_categ_data = pd.read_csv(path_to_data + '/item_categories.csv')
    # Translations from google translate
    item_categ_names_eng = pd.read_table('item_categories_names_translated.txt', header=None)

    item_categ_data['item_category_name_eng'] = item_categ_names_eng
    item_categ_data['item_category_group'] = (item_categ_data['item_category_name_eng']
                                              .apply(lambda x: x.split(' - ')[0])
                                              .astype('category'))
    return item_categ_data


def clean_shops_data():
    shops_data = pd.read_csv(path_to_data + '/shops.csv')
    # Translations from google translate
    shop_names_eng = pd.read_table('shops_translated.txt', header=None)
    shops_data['shop_name_eng'] = shop_names_eng

    # standardize shop names to extract further info
    shops_data['shop_name_eng'] = shops_data['shop_name_eng'].apply(lambda x: x.lower())
    shops_data['shop_city'] = (shops_data['shop_name_eng'].apply(lambda x: x.split(' ')[0])
                               .astype('category'))
    return shops_data


def merge_all_train_data():
    sales_train_data = clean_sales_data()
    items_data = clean_items_data()
    item_categ_data = clean_items_categ_data()
    shops_data = clean_shops_data()

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
