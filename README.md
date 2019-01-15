For Kaggle use

フォルダ構造は以下の通り.
```
.
├── README.md
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── *.csv
│   │   └── *.feather
│   └── output
├── features
│   └── *.feather
├── logs
│   └── logger.py
├── models
│   └── lgbmClassifier.py
├── notebook
├── run.py
├── scripts
│   ├── convert_to_feather.py
│   ├── create_features.py
│   └── features_base.py
└── utils
    └── __init__.py
```

初めは, 以下で train.csv, test.csv を feather に変換する.
```
python scripts/convert_to_feather.py
```
そして, 特徴量の feather化 を以下で行う.
```
python scripts/create_features.py
```
