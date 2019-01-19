# フォルダ構造は以下の通り
```
.
├── README.md
├── convert_to_feather.py
├── data
│   ├── input
│   │   ├── *.csv
│   │   └── *.feather
│   └── output
│   │   └── *
├── features
│   └── *.feather
├── logs
│   └── *.log
├── notebook
│   └── *.ipynb
├── src
│   └── features_base.py
├── user01
│   ├── create_features.py
│   ├── models
│   │   ├── kfold_lgbm.py
│   │   └── kfold_xgb.py
│   └── run.py
├── user02
│   ├── create_features.py
│   ├── models
│   │   ├── kfold_lgbm.py
│   │   └── kfold_xgb.py
│   └── run.py
└── utils
    └── __init__.py
```

# 使い方
初めは, 以下で train.csv, test.csv を feather に変換する.(基本的に一度きり)
```
python convert_to_feather.py
```
### ユーザーごとの作業は基本 ./user で行う(user01とuser02は全く同じ内容)
特徴量の feather化 を以下で行う.(特徴量に変更があった時は, 該当する特徴量ファイル（/features/*.feather）を削除の上で, 再実行.あるいは別のクラスを定義して実行)
```
python create_features.py
```
学習モデルの中身は kfold_lgbm.py の中をいじってみてください.
全体を実行するのは, 以下のみ(この中から上記の create_features.py をkickすることも可能).
```
python run.py
```

# TODO
utils の make_output_dir の作りが適当（変な名前なフォルダがあると死ぬ）  
create_features.py にデバッグ機能を追加したい
