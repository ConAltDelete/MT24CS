# Pipeline idea

- Data -> `Data_visual_cleaning.py` -> `modeling.py` -> `plots/*.pdf` -> `thesis.pdf`
- `Data_visual_cleaning.py` := Data -> `plot_diff()` -> `remove_outliers()` -> `impute_Nan(limit = n_days)` -> `save_to_bin()` -> Data.bin
- `modeling.py` := Data | Data.bin -> `Model_A()` & `Model_B()` & `Model_C()` & ... & `Model_END()` -> `plot_result()` -> `Stat_of_model()`
- `thesis.pdf` := plots & stat & analesys -> `thesis.tex` -> `thesis.pdf`