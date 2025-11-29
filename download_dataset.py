import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('bertvankeulen/cicids-2017', path = r'D:\ids_ml_course\data\raw', unzip=True)
