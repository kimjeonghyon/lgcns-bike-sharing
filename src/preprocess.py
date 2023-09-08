import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "season",
]

preprocess_pipeline = ColumnTransformer(
    transformers=[
        ("target_encoder", TargetEncoder(cols=CAT_FEATURES), CAT_FEATURES),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
