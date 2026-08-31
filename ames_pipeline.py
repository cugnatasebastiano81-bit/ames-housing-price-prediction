"""Pipeline riproducibile per il dataset Ames Housing.

Il modulo contiene la logica testabile usata dal notebook canonico:

* caricamento e validazione del dataset;
* split sviluppo/test riproducibile;
* wrangling appreso esclusivamente durante ``fit``;
* feature engineering e preprocessing sklearn;
* selezione tramite cross-validation senza accesso al test;
* valutazione del singolo modello congelato;
* contratto stretto per ``predict_price(input_dict)``.

Il test set non compare in alcuna funzione di selezione. La funzione di valutazione
accetta soltanto un modello già selezionato e già addestrato sull'insieme di sviluppo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42
TARGET = "SalePrice"
ID_COLS = ("Order", "PID")
EXPECTED_DATASET_SHA256 = "65A1CBB89C2B58B11674097135DD04E710DF5DB726BCFE6EC551DE38B44E4911"
EXPECTED_DATASET_ROWS = 2930

EXPECTED_COLUMNS = (
    "Order",
    "PID",
    "MS SubClass",
    "MS Zoning",
    "Lot Frontage",
    "Lot Area",
    "Street",
    "Alley",
    "Lot Shape",
    "Land Contour",
    "Utilities",
    "Lot Config",
    "Land Slope",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Mas Vnr Area",
    "Exter Qual",
    "Exter Cond",
    "Foundation",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin SF 1",
    "BsmtFin Type 2",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Heating",
    "Heating QC",
    "Central Air",
    "Electrical",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "Kitchen Qual",
    "TotRms AbvGrd",
    "Functional",
    "Fireplaces",
    "Fireplace Qu",
    "Garage Type",
    "Garage Yr Blt",
    "Garage Finish",
    "Garage Cars",
    "Garage Area",
    "Garage Qual",
    "Garage Cond",
    "Paved Drive",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
    "Sale Type",
    "Sale Condition",
    "SalePrice",
)

RAW_FEATURE_COLUMNS = tuple(
    column for column in EXPECTED_COLUMNS if column not in {*ID_COLS, TARGET}
)

SEMANTIC_NONE_CAT = (
    "Alley",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Mas Vnr Type",
)

STRUCTURAL_ZERO_NUM = (
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Garage Cars",
    "Garage Area",
)

ORDINAL_MAPS = {
    "Exter Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Exter Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Heating QC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Kitchen Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Qual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Cond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Fireplace Qu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Qual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Cond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Pool QC": ["None", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Exposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFin Type 1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFin Type 2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Garage Finish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Lot Shape": ["IR3", "IR2", "IR1", "Reg"],
    "Land Slope": ["Sev", "Mod", "Gtl"],
    "Paved Drive": ["N", "P", "Y"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}

ORDINAL_COLUMNS = tuple(ORDINAL_MAPS)
NOMINAL_COLUMNS = (
    "MS Zoning",
    "Street",
    "Alley",
    "Land Contour",
    "Lot Config",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Foundation",
    "Heating",
    "Central Air",
    "Electrical",
    "Garage Type",
    "Fence",
    "Misc Feature",
    "Sale Type",
    "Sale Condition",
)

CATEGORICAL_COLUMNS = ORDINAL_COLUMNS + NOMINAL_COLUMNS
NUMERIC_RAW_COLUMNS = tuple(
    column
    for column in RAW_FEATURE_COLUMNS
    if column not in set(CATEGORICAL_COLUMNS)
)
INFERENCE_INTEGER_COLUMNS = frozenset(
    {
        "Overall Qual",
        "Overall Cond",
        "Mo Sold",
        "Year Built",
        "Year Remod/Add",
        "Yr Sold",
        "Garage Yr Blt",
    }
)
ENGINEERED_NUMERIC_COLUMNS = (
    "House Age",
    "Remod Age",
    "Is Remodeled",
    "Garage Age",
    "Total SF",
    "Total Porch SF",
    "Total Bathrooms",
    "Area per Room",
    "Overall Score",
    "Qual x Area",
    "Has Pool",
    "Has Fireplace",
    "Has 2nd Floor",
    "Has Bsmt",
)
NUMERIC_WRANGLED_COLUMNS = NUMERIC_RAW_COLUMNS + ("Has Garage",)

SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}
COMPLEXITY_ORDER = (
    "LinearRegression",
    "Ridge",
    "RandomForest",
    "XGBoost",
)

_FINAL_PIPELINE: Pipeline | None = None


def sha256_file(path: str | Path) -> str:
    """Restituisce lo SHA-256 maiuscolo di un file senza modificarlo."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_dataset(df: pd.DataFrame, *, expected_rows: int = EXPECTED_DATASET_ROWS) -> None:
    """Valida forma, schema, identificatori e target del dataset canonico."""

    if tuple(df.columns) != EXPECTED_COLUMNS:
        missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
        extra = [column for column in df.columns if column not in EXPECTED_COLUMNS]
        raise ValueError(
            "Schema Ames non valido. "
            f"Colonne mancanti: {missing or 'nessuna'}; "
            f"colonne inattese: {extra or 'nessuna'}; "
            "anche l'ordine deve coincidere con lo schema canonico."
        )
    if len(df) != expected_rows:
        raise ValueError(f"Righe Ames attese: {expected_rows}; osservate: {len(df)}.")

    for identifier in ID_COLS:
        if df[identifier].isna().any():
            raise ValueError(f"L'identificatore {identifier!r} contiene valori mancanti.")
        if df[identifier].duplicated().any():
            raise ValueError(f"L'identificatore {identifier!r} contiene duplicati.")

    target = pd.to_numeric(df[TARGET], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("SalePrice deve contenere soltanto valori numerici finiti.")
    if (target <= 0).any():
        raise ValueError("SalePrice deve essere strettamente positivo.")

    for column in NUMERIC_RAW_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"La colonna numerica {column!r} contiene valori non numerici.")


def load_data(
    path: str | Path = "AmesHousing.csv",
    *,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """Carica il CSV canonico preservando il significato della stringa ``NA``."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"File non trovato: {data_path}")
    if verify_hash:
        observed_hash = sha256_file(data_path)
        if observed_hash != EXPECTED_DATASET_SHA256:
            raise ValueError(
                "Hash del dataset non valido. "
                f"Atteso {EXPECTED_DATASET_SHA256}; osservato {observed_hash}."
            )

    df = pd.read_csv(
        data_path,
        keep_default_na=False,
        na_values=[""],
        dtype={"Order": "string", "PID": "string"},
    )
    df.columns = df.columns.str.strip()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    df[text_columns] = df[text_columns].apply(lambda series: series.str.strip())
    validate_dataset(df)
    return df


def split_development_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide i dati prima di ogni EDA o statistica appresa."""

    development, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return development.reset_index(drop=True), test.reset_index(drop=True)


def split_manifest_digest(df: pd.DataFrame) -> str:
    """Digest stabile degli identificatori che compongono uno split."""

    rows = sorted(
        f"{str(order)}|{str(pid)}"
        for order, pid in zip(df["Order"], df["PID"], strict=True)
    )
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return sha256(payload).hexdigest().upper()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa le 79 feature grezze dal target trasformato con ``log1p``."""

    missing = [column for column in (*RAW_FEATURE_COLUMNS, TARGET) if column not in df.columns]
    if missing:
        raise ValueError(f"Colonne necessarie mancanti: {missing}")
    features = df.loc[:, RAW_FEATURE_COLUMNS].copy()
    target = np.log1p(pd.to_numeric(df[TARGET], errors="raise")).rename(TARGET)
    return features, target


def _require_raw_feature_frame(X: Any) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("La pipeline Ames richiede un pandas.DataFrame con colonne nominate.")
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in X.columns]
    if missing:
        raise ValueError(f"Feature Ames mancanti: {missing}")
    return X.loc[:, RAW_FEATURE_COLUMNS].copy()


def _apply_deterministic_fixes(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()

    for column in SEMANTIC_NONE_CAT:
        result[column] = result[column].replace("NA", "None").fillna("None")

    for column in NUMERIC_RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in STRUCTURAL_ZERO_NUM:
        result[column] = result[column].fillna(0)

    garage_year = pd.to_numeric(result["Garage Yr Blt"], errors="coerce")
    result["Garage Yr Blt"] = garage_year.mask(garage_year == 2207, 2007)
    return result


class AmesWrangler(BaseEstimator, TransformerMixin):
    """Wrangling stateful compatibile con la cross-validation sklearn.

    Le mediane di ``Lot Frontage`` e la moda di ``Electrical`` vengono apprese
    durante ``fit``. Poiché il transformer è il primo step della pipeline, ogni
    fold apprende questi valori esclusivamente dalle proprie righe di training.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> AmesWrangler:
        frame = _apply_deterministic_fixes(_require_raw_feature_frame(X))
        frontage = pd.to_numeric(frame["Lot Frontage"], errors="coerce")
        medians = frontage.groupby(frame["Neighborhood"]).median().dropna()
        global_median = frontage.median()
        if pd.isna(global_median):
            raise ValueError("Impossibile apprendere la mediana globale di Lot Frontage.")

        electrical = frame["Electrical"].replace("NA", np.nan).dropna()
        if electrical.empty:
            raise ValueError("Impossibile apprendere la moda di Electrical.")

        self.lot_frontage_medians_ = medians.to_dict()
        self.lot_frontage_global_ = float(global_median)
        self.electrical_mode_ = str(electrical.mode().iloc[0])
        self.feature_names_in_ = np.asarray(RAW_FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "lot_frontage_medians_",
                "lot_frontage_global_",
                "electrical_mode_",
            ],
        )
        frame = _apply_deterministic_fixes(_require_raw_feature_frame(X))
        fill_values = (
            frame["Neighborhood"]
            .map(self.lot_frontage_medians_)
            .fillna(self.lot_frontage_global_)
        )
        frame["Lot Frontage"] = pd.to_numeric(
            frame["Lot Frontage"], errors="coerce"
        ).fillna(fill_values)
        frame["Electrical"] = (
            frame["Electrical"].replace("NA", np.nan).fillna(self.electrical_mode_)
        )
        frame["Has Garage"] = (frame["Garage Type"] != "None").astype(int)
        frame["Garage Yr Blt"] = pd.to_numeric(
            frame["Garage Yr Blt"], errors="coerce"
        ).fillna(0)
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_in_"])
        return np.asarray((*RAW_FEATURE_COLUMNS, "Has Garage"), dtype=object)


def wrangle(
    df: pd.DataFrame,
    *,
    transformer: AmesWrangler | None = None,
) -> tuple[pd.DataFrame, AmesWrangler]:
    """Wrapper didattico che evita una seconda implementazione del wrangling."""

    features = df.drop(columns=[TARGET, *ID_COLS], errors="ignore")
    if transformer is None:
        transformer = AmesWrangler().fit(features)
    return transformer.transform(features), transformer


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea le feature Ames derivate senza mutare il DataFrame in ingresso."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer richiede un pandas.DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_in_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer richiede un pandas.DataFrame.")
        result = X.copy()
        if not self.enabled:
            return result

        result["House Age"] = (result["Yr Sold"] - result["Year Built"]).clip(lower=0)
        result["Remod Age"] = (
            result["Yr Sold"] - result["Year Remod/Add"]
        ).clip(lower=0)
        result["Is Remodeled"] = (
            result["Year Built"] != result["Year Remod/Add"]
        ).astype(int)
        result["Garage Age"] = np.where(
            result["Garage Yr Blt"] > 0,
            (result["Yr Sold"] - result["Garage Yr Blt"]).clip(lower=0),
            0,
        )
        result["Total SF"] = (
            result["Total Bsmt SF"] + result["1st Flr SF"] + result["2nd Flr SF"]
        )
        result["Total Porch SF"] = (
            result["Open Porch SF"]
            + result["Enclosed Porch"]
            + result["3Ssn Porch"]
            + result["Screen Porch"]
            + result["Wood Deck SF"]
        )
        result["Total Bathrooms"] = (
            result["Full Bath"]
            + 0.5 * result["Half Bath"]
            + result["Bsmt Full Bath"]
            + 0.5 * result["Bsmt Half Bath"]
        )
        rooms = result["TotRms AbvGrd"].replace(0, np.nan)
        result["Area per Room"] = (result["Gr Liv Area"] / rooms).fillna(0)
        result["Overall Score"] = result["Overall Qual"] * result["Overall Cond"]
        result["Qual x Area"] = result["Overall Qual"] * result["Gr Liv Area"]
        result["Has Pool"] = (result["Pool Area"] > 0).astype(int)
        result["Has Fireplace"] = (result["Fireplaces"] > 0).astype(int)
        result["Has 2nd Floor"] = (result["2nd Flr SF"] > 0).astype(int)
        result["Has Bsmt"] = (result["Total Bsmt SF"] > 0).astype(int)
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_in_"])
        if not self.enabled:
            return self.feature_names_in_.copy()
        return np.asarray((*self.feature_names_in_, *ENGINEERED_NUMERIC_COLUMNS), dtype=object)


class ConditionalLogTransformer(BaseEstimator, TransformerMixin):
    """Applica ``log1p`` alle colonne non negative con skew sopra soglia."""

    def __init__(self, skew_threshold: float = 0.75):
        self.skew_threshold = skew_threshold

    def fit(self, X: Any, y: Any = None) -> ConditionalLogTransformer:
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
        skew = frame.apply(lambda series: series.skew())
        self.transform_mask_ = (
            (skew.abs() > self.skew_threshold) & (frame.min() >= 0)
        ).to_numpy(dtype=bool)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["transform_mask_", "n_features_in_"])
        values = np.asarray(X, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Attese {self.n_features_in_} feature numeriche; osservata forma {values.shape}."
            )
        for index, should_transform in enumerate(self.transform_mask_):
            if should_transform:
                if np.any(values[:, index] < 0):
                    raise ValueError("log1p non è ammesso su valori negativi in inference.")
                values[:, index] = np.log1p(values[:, index])
        return values

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        if input_features is None:
            return np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)], dtype=object
            )
        return np.asarray(input_features, dtype=object)


def build_preprocessor(*, with_feature_engineering: bool = True) -> ColumnTransformer:
    numeric_columns = list(NUMERIC_WRANGLED_COLUMNS)
    if with_feature_engineering:
        numeric_columns.extend(ENGINEERED_NUMERIC_COLUMNS)

    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_MAPS[column] for column in ORDINAL_COLUMNS],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        min_frequency=10,
    )
    numeric_pipeline = Pipeline(
        [
            ("conditional_log", ConditionalLogTransformer(skew_threshold=0.75)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "ordinal",
                Pipeline(
                    [
                        ("encoder", ordinal_encoder),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(ORDINAL_COLUMNS),
            ),
            ("nominal", onehot_encoder, list(NOMINAL_COLUMNS)),
            ("numeric", numeric_pipeline, numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    *,
    with_feature_engineering: bool = True,
) -> Pipeline:
    """Crea una pipeline completa raw-input -> previsione logaritmica."""

    return Pipeline(
        [
            ("wrangle", AmesWrangler()),
            ("feature_engineering", FeatureEngineer(enabled=with_feature_engineering)),
            (
                "preprocess",
                build_preprocessor(with_feature_engineering=with_feature_engineering),
            ),
            ("model", estimator),
        ]
    )


def shared_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materializza gli stessi fold da riusare per ogni candidato."""

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kfold.split(X, y))


def _singleton_configs(configs: Sequence[Mapping[str, Any]]) -> list[dict[str, list[Any]]]:
    return [{name: [value] for name, value in config.items()} for config in configs]


def _load_xgb_regressor() -> type[BaseEstimator]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost non è disponibile nell'ambiente corrente. "
            "Installare la dipendenza fissata per usare i modelli XGBoost."
        ) from exc
    return XGBRegressor


def build_tuned_searches(
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, RandomizedSearchCV]:
    """Costruisce inventari finiti e riproducibili di configurazioni confrontabili."""

    XGBRegressor = _load_xgb_regressor()

    ridge_configs = [{"model__alpha": value} for value in (0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0)]
    rf_configs = [
        {
            "model__n_estimators": 120,
            "model__max_depth": 20,
            "model__max_features": 1.0,
            "model__min_samples_leaf": 1,
        },
        {
            "model__n_estimators": 160,
            "model__max_depth": 15,
            "model__max_features": 0.5,
            "model__min_samples_leaf": 2,
        },
        {
            "model__n_estimators": 200,
            "model__max_depth": 20,
            "model__max_features": 0.5,
            "model__min_samples_leaf": 1,
        },
        {
            "model__n_estimators": 200,
            "model__max_depth": 25,
            "model__max_features": "sqrt",
            "model__min_samples_leaf": 1,
        },
        {
            "model__n_estimators": 160,
            "model__max_depth": None,
            "model__max_features": 0.3,
            "model__min_samples_leaf": 2,
        },
        {
            "model__n_estimators": 200,
            "model__max_depth": None,
            "model__max_features": 1.0,
            "model__min_samples_leaf": 2,
        },
    ]
    xgb_configs = [
        {
            "model__n_estimators": 500,
            "model__learning_rate": 0.05,
            "model__max_depth": 4,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 1.0,
        },
        {
            "model__n_estimators": 400,
            "model__learning_rate": 0.05,
            "model__max_depth": 3,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 1.0,
        },
        {
            "model__n_estimators": 600,
            "model__learning_rate": 0.03,
            "model__max_depth": 3,
            "model__subsample": 0.8,
            "model__colsample_bytree": 1.0,
            "model__reg_lambda": 1.0,
        },
        {
            "model__n_estimators": 600,
            "model__learning_rate": 0.05,
            "model__max_depth": 4,
            "model__subsample": 0.7,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 2.0,
        },
        {
            "model__n_estimators": 800,
            "model__learning_rate": 0.02,
            "model__max_depth": 4,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.7,
            "model__reg_lambda": 1.0,
        },
        {
            "model__n_estimators": 800,
            "model__learning_rate": 0.03,
            "model__max_depth": 5,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 2.0,
        },
        {
            "model__n_estimators": 600,
            "model__learning_rate": 0.1,
            "model__max_depth": 3,
            "model__subsample": 1.0,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 0.5,
        },
        {
            "model__n_estimators": 400,
            "model__learning_rate": 0.1,
            "model__max_depth": 4,
            "model__subsample": 1.0,
            "model__colsample_bytree": 1.0,
            "model__reg_lambda": 1.0,
        },
    ]

    search_specs = {
        "Ridge": (
            build_model_pipeline(Ridge()),
            ridge_configs,
        ),
        "RandomForest": (
            build_model_pipeline(
                RandomForestRegressor(random_state=random_state, n_jobs=1)
            ),
            rf_configs,
        ),
        "XGBoost": (
            build_model_pipeline(
                XGBRegressor(
                    random_state=random_state,
                    tree_method="hist",
                    n_jobs=1,
                    importance_type="gain",
                )
            ),
            xgb_configs,
        ),
    }

    searches: dict[str, RandomizedSearchCV] = {}
    for name, (pipeline, configs) in search_specs.items():
        distributions = _singleton_configs(configs)
        searches[name] = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=distributions,
            n_iter=len(distributions),
            scoring=SCORING,
            refit="RMSE",
            cv=list(cv_splits),
            random_state=random_state,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
    return searches


def select_by_one_standard_error(
    table: pd.DataFrame,
    *,
    n_splits: int,
    complexity_order: Sequence[str] = COMPLEXITY_ORDER,
) -> tuple[str, float]:
    """Applica la regola di parsimonia senza consultare metriche del test."""

    required = {"RMSE", "RMSE_std"}
    if not required.issubset(table.columns):
        raise ValueError(f"La tabella deve contenere {sorted(required)}.")
    if table.empty:
        raise ValueError("La tabella di selezione è vuota.")

    best_name = str(table["RMSE"].idxmin())
    best_mean = float(table.loc[best_name, "RMSE"])
    best_standard_error = float(table.loc[best_name, "RMSE_std"]) / np.sqrt(n_splits)
    threshold = best_mean + best_standard_error
    eligible = set(table.index[table["RMSE"] <= threshold])

    for name in complexity_order:
        if name in eligible:
            return name, threshold
    return best_name, threshold


@dataclass
class SelectionResult:
    selected_name: str
    selected_estimator: Pipeline
    table: pd.DataFrame
    one_standard_error_threshold: float
    fitted_candidates: dict[str, Pipeline]


def run_model_selection(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    *,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> SelectionResult:
    """Confronta quattro candidati usando esclusivamente l'insieme di sviluppo."""

    if cv_splits is None:
        cv_splits = shared_cv_splits(X_development, y_development)
    cv_splits = list(cv_splits)

    rows: list[dict[str, float | str]] = []
    fitted_candidates: dict[str, Pipeline] = {}

    linear = build_model_pipeline(LinearRegression())
    linear_cv = cross_validate(
        linear,
        X_development,
        y_development,
        cv=cv_splits,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    rows.append(
        {
            "Modello": "LinearRegression",
            "RMSE": -float(np.mean(linear_cv["test_RMSE"])),
            "RMSE_std": float(np.std(linear_cv["test_RMSE"])),
            "MAE": -float(np.mean(linear_cv["test_MAE"])),
            "R2": float(np.mean(linear_cv["test_R2"])),
        }
    )
    fitted_candidates["LinearRegression"] = linear.fit(X_development, y_development)

    for name, search in build_tuned_searches(cv_splits).items():
        search.fit(X_development, y_development)
        index = int(search.best_index_)
        results = search.cv_results_
        rows.append(
            {
                "Modello": name,
                "RMSE": -float(results["mean_test_RMSE"][index]),
                "RMSE_std": float(results["std_test_RMSE"][index]),
                "MAE": -float(results["mean_test_MAE"][index]),
                "R2": float(results["mean_test_R2"][index]),
            }
        )
        fitted_candidates[name] = search.best_estimator_

    table = pd.DataFrame(rows).set_index("Modello").sort_values("RMSE")
    selected_name, threshold = select_by_one_standard_error(
        table,
        n_splits=len(cv_splits),
    )
    return SelectionResult(
        selected_name=selected_name,
        selected_estimator=fitted_candidates[selected_name],
        table=table,
        one_standard_error_threshold=threshold,
        fitted_candidates=fitted_candidates,
    )


def feature_ablation_cv(
    fitted_pipeline: Pipeline,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Confronta con/senza feature derivate sugli stessi fold di sviluppo."""

    estimator = clone(fitted_pipeline.named_steps["model"])
    variants = {
        "Con feature derivate": build_model_pipeline(
            clone(estimator), with_feature_engineering=True
        ),
        "Senza feature derivate": build_model_pipeline(
            clone(estimator), with_feature_engineering=False
        ),
    }
    fold_scores: dict[str, np.ndarray] = {}
    for name, pipeline in variants.items():
        result = cross_validate(
            pipeline,
            X_development,
            y_development,
            cv=list(cv_splits),
            scoring={"RMSE": SCORING["RMSE"]},
            n_jobs=1,
            error_score="raise",
        )
        fold_scores[name] = -np.asarray(result["test_RMSE"], dtype=float)

    output = pd.DataFrame(fold_scores)
    output["Delta (con - senza)"] = (
        output["Con feature derivate"] - output["Senza feature derivate"]
    )
    output.index = [f"Fold {index}" for index in range(1, len(output) + 1)]
    return output


def evaluate_holdout(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Valuta un solo modello già congelato sul test e restituisce residui."""

    check_is_fitted(fitted_pipeline)
    prediction_log = np.asarray(fitted_pipeline.predict(X_test), dtype=float)
    truth_log = np.asarray(y_test_log, dtype=float)
    truth_dollars = np.expm1(truth_log)
    prediction_dollars = np.expm1(prediction_log)

    if not np.isfinite(prediction_dollars).all() or np.any(prediction_dollars <= 0):
        raise RuntimeError("Il modello ha prodotto almeno un prezzo non valido.")

    metrics = pd.DataFrame(
        [
            {
                "RMSE log": np.sqrt(mean_squared_error(truth_log, prediction_log)),
                "MAE log": mean_absolute_error(truth_log, prediction_log),
                "R2 log": r2_score(truth_log, prediction_log),
                "RMSE $": np.sqrt(
                    mean_squared_error(truth_dollars, prediction_dollars)
                ),
                "MAE $": mean_absolute_error(truth_dollars, prediction_dollars),
                "MAPE %": np.mean(
                    np.abs((truth_dollars - prediction_dollars) / truth_dollars)
                )
                * 100,
            }
        ]
    )
    residuals = truth_log - prediction_log
    return metrics, prediction_log, residuals


def transformed_feature_importance(fitted_pipeline: Pipeline) -> pd.Series:
    """Restituisce gain o coefficienti assoluti con i nomi delle feature trasformate."""

    check_is_fitted(fitted_pipeline)
    transformer = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    names = transformer.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_).astype(float))
    else:
        raise TypeError("Il modello non espone feature_importances_ o coef_.")
    if len(names) != len(values):
        raise RuntimeError("Nomi e importanze delle feature hanno lunghezze diverse.")
    return pd.Series(values, index=names).sort_values(ascending=False)


def _is_missing_scalar(value: Any) -> bool:
    if isinstance(value, list | tuple | dict | set | np.ndarray | pd.Series):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_inference_input(input_dict: Mapping[str, Any]) -> pd.DataFrame:
    """Valida e normalizza una singola osservazione secondo il contratto pubblico."""

    if not isinstance(input_dict, Mapping):
        raise TypeError("input_dict deve implementare collections.abc.Mapping.")

    non_string_key_types = sorted(
        {type(key).__name__ for key in input_dict if not isinstance(key, str)}
    )
    if non_string_key_types:
        raise ValueError(
            "Le chiavi di input devono essere stringhe; "
            f"tipi non validi: {non_string_key_types}."
        )

    keys = set(input_dict)
    forbidden = sorted(keys.intersection({*ID_COLS, TARGET}))
    if forbidden:
        raise ValueError(f"Campi vietati in inference: {forbidden}")
    missing = sorted(set(RAW_FEATURE_COLUMNS) - keys)
    extra = sorted(keys - set(RAW_FEATURE_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"Schema input non valido. Mancanti: {missing or 'nessuno'}; "
            f"inattesi: {extra or 'nessuno'}."
        )

    normalized: dict[str, Any] = {}
    allowed_missing_numeric = {*STRUCTURAL_ZERO_NUM, "Lot Frontage", "Garage Yr Blt"}

    for column in RAW_FEATURE_COLUMNS:
        value = input_dict[column]
        if column in NUMERIC_RAW_COLUMNS:
            if isinstance(value, bool | np.bool_):
                raise ValueError(f"{column}: un booleano non è un valore numerico valido.")
            if _is_missing_scalar(value):
                if column not in allowed_missing_numeric:
                    raise ValueError(f"{column}: valore mancante non ammesso.")
                normalized[column] = np.nan
                continue
            if not isinstance(value, int | float | np.integer | np.floating):
                raise ValueError(
                    f"{column}: è richiesto un numero scalare Python o NumPy, "
                    f"ricevuto {type(value).__name__}."
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column}: valore numerico non valido {value!r}.") from exc
            if not np.isfinite(number):
                raise ValueError(f"{column}: NaN e infinito non sono ammessi.")
            if number < 0:
                raise ValueError(f"{column}: i valori negativi non sono ammessi.")
            if column in INFERENCE_INTEGER_COLUMNS and not number.is_integer():
                raise ValueError(f"{column}: è richiesto un valore intero.")
            normalized[column] = number
            continue

        if _is_missing_scalar(value):
            if column in SEMANTIC_NONE_CAT:
                normalized[column] = "None"
            elif column == "Electrical":
                normalized[column] = np.nan
            else:
                raise ValueError(f"{column}: valore mancante non ammesso.")
            continue
        if not isinstance(value, str):
            raise ValueError(f"{column}: è richiesta una stringa, ricevuto {type(value).__name__}.")
        text = value.strip()
        if not text:
            raise ValueError(f"{column}: stringa vuota non ammessa.")
        if column in SEMANTIC_NONE_CAT and text == "NA":
            text = "None"
        if column in ORDINAL_MAPS and text not in ORDINAL_MAPS[column]:
            raise ValueError(
                f"{column}: categoria ordinale sconosciuta {text!r}; "
                f"valori ammessi: {ORDINAL_MAPS[column]}."
            )
        normalized[column] = text

    if not 1 <= normalized["Overall Qual"] <= 10:
        raise ValueError("Overall Qual deve essere compreso fra 1 e 10.")
    if not 1 <= normalized["Overall Cond"] <= 10:
        raise ValueError("Overall Cond deve essere compreso fra 1 e 10.")
    if not 1 <= normalized["Mo Sold"] <= 12:
        raise ValueError("Mo Sold deve essere compreso fra 1 e 12.")

    year_built = normalized["Year Built"]
    year_remodeled = normalized["Year Remod/Add"]
    year_sold = normalized["Yr Sold"]
    if not year_built <= year_remodeled <= year_sold:
        raise ValueError("Gli anni devono rispettare Year Built <= Year Remod/Add <= Yr Sold.")

    garage_type = normalized["Garage Type"]
    garage_year = normalized["Garage Yr Blt"]
    garage_cars = normalized["Garage Cars"]
    garage_area = normalized["Garage Area"]
    if garage_type == "None":
        garage_values = (
            ("Garage Yr Blt", garage_year),
            ("Garage Cars", garage_cars),
            ("Garage Area", garage_area),
        )
        for label, value in garage_values:
            if not _is_missing_scalar(value) and float(value) != 0:
                raise ValueError(f"{label} deve essere 0 o mancante quando Garage Type='None'.")
    else:
        if _is_missing_scalar(garage_year) or not 1800 <= float(garage_year) <= year_sold:
            raise ValueError(
                "Garage Yr Blt deve essere un anno valido non futuro quando il garage esiste."
            )

    if normalized["Pool Area"] > 0 and normalized["Pool QC"] == "None":
        raise ValueError("Pool QC non può essere 'None' quando Pool Area è positiva.")
    if normalized["Fireplaces"] > 0 and normalized["Fireplace Qu"] == "None":
        raise ValueError("Fireplace Qu non può essere 'None' quando Fireplaces è positivo.")
    if normalized["Total Bsmt SF"] > 0 and normalized["Bsmt Qual"] == "None":
        raise ValueError("Bsmt Qual non può essere 'None' quando Total Bsmt SF è positiva.")

    return pd.DataFrame([normalized], columns=RAW_FEATURE_COLUMNS)


def set_final_pipeline(fitted_pipeline: Pipeline) -> None:
    """Registra una pipeline Ames fitted per l'inferenza simulata.

    Il codice verifica il tipo della pipeline, la struttura Ames, la famiglia del
    modello finale e lo stato fitted. Il chiamante deve fornire il candidato
    congelato già selezionato e valutato secondo il protocollo del progetto: tali
    passaggi precedenti non possono essere verificati retroattivamente dal setter.
    """

    if not isinstance(fitted_pipeline, Pipeline):
        raise TypeError("Il modello finale deve essere una sklearn Pipeline Ames.")

    expected_step_names = ("wrangle", "feature_engineering", "preprocess", "model")
    observed_step_names = tuple(name for name, _ in fitted_pipeline.steps)
    if observed_step_names != expected_step_names:
        raise ValueError(
            "La pipeline Ames deve contenere esattamente gli step "
            f"{expected_step_names!r} nell'ordine previsto; ricevuti {observed_step_names!r}."
        )

    expected_structural_types = (
        ("wrangle", AmesWrangler),
        ("feature_engineering", FeatureEngineer),
        ("preprocess", ColumnTransformer),
    )
    for step_name, expected_type in expected_structural_types:
        if not isinstance(fitted_pipeline.named_steps[step_name], expected_type):
            raise ValueError(
                f"Lo step {step_name!r} deve essere di tipo {expected_type.__name__}."
            )

    final_model = fitted_pipeline.named_steps["model"]
    supported_model = isinstance(final_model, LinearRegression | Ridge | RandomForestRegressor)
    is_xgboost_family = any(
        base.__name__ == "XGBRegressor"
        and base.__module__.partition(".")[0] == "xgboost"
        for base in type(final_model).__mro__
    )
    if not supported_model and is_xgboost_family:
        supported_model = isinstance(final_model, _load_xgb_regressor())
    if not supported_model:
        raise ValueError(
            "Lo step 'model' deve appartenere a una famiglia supportata: "
            "LinearRegression, Ridge, RandomForestRegressor o XGBRegressor."
        )

    for step_name in expected_step_names:
        check_is_fitted(fitted_pipeline.named_steps[step_name])
    global _FINAL_PIPELINE
    _FINAL_PIPELINE = fitted_pipeline


def predict_price(input_dict: Mapping[str, Any]) -> float:
    """Valida un immobile Ames e restituisce il prezzo previsto in dollari."""

    if _FINAL_PIPELINE is None:
        raise RuntimeError(
            "Nessun modello finale registrato. Chiamare set_final_pipeline() "
            "solo dopo selezione e valutazione del candidato congelato."
        )
    row = validate_inference_input(input_dict)
    prediction_log = float(np.asarray(_FINAL_PIPELINE.predict(row), dtype=float)[0])
    price = float(np.expm1(prediction_log))
    if not np.isfinite(price) or price <= 0:
        raise RuntimeError(f"Predizione non valida prodotta dal modello: {price!r}.")
    return price


__all__ = [
    "AmesWrangler",
    "COMPLEXITY_ORDER",
    "ConditionalLogTransformer",
    "ENGINEERED_NUMERIC_COLUMNS",
    "EXPECTED_COLUMNS",
    "EXPECTED_DATASET_ROWS",
    "EXPECTED_DATASET_SHA256",
    "FeatureEngineer",
    "ID_COLS",
    "ORDINAL_MAPS",
    "RANDOM_STATE",
    "RAW_FEATURE_COLUMNS",
    "SCORING",
    "SelectionResult",
    "TARGET",
    "build_model_pipeline",
    "build_preprocessor",
    "evaluate_holdout",
    "feature_ablation_cv",
    "load_data",
    "predict_price",
    "run_model_selection",
    "select_by_one_standard_error",
    "set_final_pipeline",
    "sha256_file",
    "shared_cv_splits",
    "split_development_test",
    "split_features_target",
    "split_manifest_digest",
    "transformed_feature_importance",
    "validate_dataset",
    "validate_inference_input",
    "wrangle",
]
