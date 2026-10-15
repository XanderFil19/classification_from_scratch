def load(dataset = 'glass'):
    import os
    from pathlib import Path
    from kaggle.api.kaggle_api_extended import KaggleApi
    import pandas as pd

    if dataset == 'fashion':
        dataset_slug = 'zalando-research/fashionmnist'
        output_path = Path( './data/MINST/fashion' )
        append = None
        append_test = 'fashion-mnist_test.csv'
        append_train = 'fashion-mnist_train.csv'

    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_path, exist_ok=True)

    api.dataset_download_files(dataset_slug, path=output_path, unzip=True)
    print("Dataset succesfully downloaded!")

    if append is None:
        df_train = pd.read_csv( output_path / append_train )
        df_test = pd.read_csv( output_path / append_test )
        return df_train , df_test
    
    df = pd.read_csv( output_path / append )

    return df

def augment_minority(X,y,rate):
    import numpy as np
    import pandas as pd

    c = np.argmin(y.value_counts())
    print(c)
    idx = y[ y == c ].index

    y_minority = y.iloc[idx]
    X_minority = X.iloc[idx,:]

    N = int( len(y_minority) * rate )

    Rows_x = []
    Rows_y = []

    for n in range(N):
        if n % 1000 == 0:
            print(f'Added element {n + 1} over {N}')
        number = np.random.randint(len(y_minority))
        idx = np.random.choice( np.arange(len(y_minority)) , number )
        row = np.mean( X_minority.iloc[idx,:] , axis = 0 )

        Rows_x.append(row)
        Rows_y.append(c)

    X = pd.concat( [ X , pd.DataFrame(Rows_x) ] )
    y = pd.concat( [ y , pd.Series(Rows_y) ] )

    shuffle_idx = np.random.permutation( len(X) )
    X_shuffle , y_shuffle = X.iloc[shuffle_idx,:], y.iloc[shuffle_idx]

    return X_shuffle , y_shuffle