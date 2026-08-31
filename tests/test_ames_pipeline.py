from __future__ import annotations

import importlib.util
import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import ames_pipeline as ames

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "AmesHousing.csv"


class AmesPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = ames.load_data(DATA_PATH)
        cls.development, cls.test = ames.split_development_test(cls.dataset)
        cls.X_development, cls.y_development = ames.split_features_target(cls.development)
        cls.valid_input = cls.X_development.iloc[300].to_dict()

    def setUp(self) -> None:
        ames._FINAL_PIPELINE = None

    def tearDown(self) -> None:
        ames._FINAL_PIPELINE = None

    def test_dataset_contract_and_hash(self) -> None:
        self.assertEqual(self.dataset.shape, (2930, 82))
        self.assertEqual(tuple(self.dataset.columns), ames.EXPECTED_COLUMNS)
        self.assertEqual(ames.sha256_file(DATA_PATH), ames.EXPECTED_DATASET_SHA256)

    def test_split_is_stable_and_has_no_overlap(self) -> None:
        self.assertEqual(self.development.shape, (2344, 82))
        self.assertEqual(self.test.shape, (586, 82))
        development_ids = set(zip(self.development["Order"], self.development["PID"], strict=True))
        test_ids = set(zip(self.test["Order"], self.test["PID"], strict=True))
        self.assertTrue(development_ids.isdisjoint(test_ids))
        self.assertEqual(
            ames.split_manifest_digest(self.test),
            "1F35ED4E8036514639F34F666960E6DE3A7097F9D87480BC7FC0267AD9C4FD48",
        )

    def test_notebook_selection_does_not_access_holdout(self) -> None:
        notebook_path = PROJECT_ROOT / "AmesHousing_Pipeline.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells_by_id = {cell["id"]: cell for cell in notebook["cells"]}
        selection_source = "".join(cells_by_id["selection"]["source"])
        for forbidden in ("X_test", "y_test", "test_raw"):
            self.assertNotIn(forbidden, selection_source)

        holdout_source = "".join(cells_by_id["holdout1"]["source"])
        self.assertIn("split_features_target(test_raw)", holdout_source)
        self.assertIn("evaluate_holdout", holdout_source)

        parameter_names = inspect.signature(ames.run_model_selection).parameters
        self.assertEqual(
            set(parameter_names),
            {"X_development", "y_development", "cv_splits"},
        )

    def test_wrangler_learns_fallback_from_fit_rows_only(self) -> None:
        training = self.X_development.iloc[:100].copy()
        wrangler = ames.AmesWrangler().fit(training)
        probe = self.X_development.iloc[[100]].copy()
        probe.loc[:, "Neighborhood"] = "Quartiere non osservato"
        probe.loc[:, "Lot Frontage"] = np.nan
        transformed = wrangler.transform(probe)
        self.assertEqual(
            float(transformed.iloc[0]["Lot Frontage"]),
            wrangler.lot_frontage_global_,
        )

    def test_feature_engineer_handles_zero_rooms_without_mutating_input(self) -> None:
        wrangled = ames.AmesWrangler().fit_transform(self.X_development.iloc[:20])
        probe = wrangled.iloc[[0]].copy()
        probe.loc[:, "TotRms AbvGrd"] = 0
        original_columns = tuple(probe.columns)
        engineer = ames.FeatureEngineer().fit(probe)
        result = engineer.transform(probe)
        self.assertEqual(float(result.iloc[0]["Area per Room"]), 0.0)
        self.assertEqual(tuple(probe.columns), original_columns)
        self.assertNotIn("Area per Room", probe.columns)

    def test_no_garage_keeps_missing_year_as_structural_zero(self) -> None:
        no_garage = self.X_development.loc[
            self.X_development["Garage Type"].eq("NA")
            & self.X_development["Garage Yr Blt"].isna()
        ]
        self.assertFalse(no_garage.empty)
        probe = no_garage.iloc[[0]].copy()

        wrangled = ames.AmesWrangler().fit(self.X_development).transform(probe)
        engineered = ames.FeatureEngineer().fit_transform(wrangled)

        observed = (
            int(wrangled.iloc[0]["Has Garage"]),
            float(wrangled.iloc[0]["Garage Yr Blt"]),
            float(engineered.iloc[0]["Garage Age"]),
        )
        self.assertEqual(observed, (0, 0.0, 0.0))

    def test_future_but_coherent_garage_year_is_preserved(self) -> None:
        probe = dict(self.valid_input)
        probe["Garage Yr Blt"] = 2015
        probe["Yr Sold"] = 2020
        validated = ames.validate_inference_input(probe)

        wrangled = ames.AmesWrangler().fit(self.X_development).transform(validated)
        engineered = ames.FeatureEngineer().fit_transform(wrangled)

        self.assertEqual(float(wrangled.iloc[0]["Garage Yr Blt"]), 2015.0)
        self.assertEqual(float(engineered.iloc[0]["Garage Age"]), 5.0)

    def test_canonical_2207_garage_year_is_corrected_to_2007(self) -> None:
        canonical_anomaly = self.dataset.loc[
            self.dataset["Order"].eq("2261") & self.dataset["PID"].eq("0916384070"),
            ames.RAW_FEATURE_COLUMNS,
        ]
        self.assertEqual(len(canonical_anomaly), 1)
        self.assertEqual(float(canonical_anomaly.iloc[0]["Garage Yr Blt"]), 2207.0)

        wrangled = ames.AmesWrangler().fit(self.X_development).transform(canonical_anomaly)

        self.assertEqual(float(wrangled.iloc[0]["Garage Yr Blt"]), 2007.0)

    def test_conditional_log_requires_fit(self) -> None:
        transformer = ames.ConditionalLogTransformer()
        with self.assertRaises(NotFittedError):
            transformer.transform(np.array([[1.0, 2.0]]))

    def test_conditional_log_does_not_log_columns_with_negative_training_values(self) -> None:
        values = np.array(
            [
                [0.0, -2.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [100.0, 1.0],
            ]
        )
        transformer = ames.ConditionalLogTransformer(skew_threshold=0.75).fit(values)
        transformed = transformer.transform(values)
        self.assertAlmostEqual(transformed[-1, 0], np.log1p(100.0))
        np.testing.assert_allclose(transformed[:, 1], values[:, 1])

    def test_one_standard_error_rule_prefers_simpler_eligible_model(self) -> None:
        table = pd.DataFrame(
            {
                "RMSE": [0.1200, 0.1161, 0.1162, 0.1300],
                "RMSE_std": [0.0040, 0.0072, 0.0092, 0.0100],
            },
            index=["LinearRegression", "Ridge", "XGBoost", "RandomForest"],
        )
        selected, threshold = ames.select_by_one_standard_error(table, n_splits=5)
        self.assertEqual(selected, "Ridge")
        self.assertGreater(threshold, table.loc["XGBoost", "RMSE"])

    def test_inference_contract_accepts_key_order_changes(self) -> None:
        reversed_input = dict(reversed(list(self.valid_input.items())))
        validated = ames.validate_inference_input(reversed_input)
        self.assertEqual(tuple(validated.columns), ames.RAW_FEATURE_COLUMNS)
        self.assertEqual(validated.shape, (1, 79))

    def test_inference_contract_rejects_missing_and_extra_fields(self) -> None:
        missing = dict(self.valid_input)
        missing.pop("Lot Area")
        with self.assertRaisesRegex(ValueError, "Mancanti"):
            ames.validate_inference_input(missing)

        extra = dict(self.valid_input)
        extra["Lot Areaa"] = extra["Lot Area"]
        with self.assertRaisesRegex(ValueError, "inattesi"):
            ames.validate_inference_input(extra)

    def test_inference_contract_rejects_target_and_identifiers(self) -> None:
        for forbidden in ("SalePrice", "Order", "PID"):
            probe = dict(self.valid_input)
            probe[forbidden] = 1
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "vietati"):
                    ames.validate_inference_input(probe)

    def test_inference_contract_rejects_unknown_ordinal(self) -> None:
        probe = dict(self.valid_input)
        probe["Exter Qual"] = "Super"
        with self.assertRaisesRegex(ValueError, "categoria ordinale sconosciuta"):
            ames.validate_inference_input(probe)

    def test_inference_contract_accepts_unknown_nominal(self) -> None:
        probe = dict(self.valid_input)
        probe["Neighborhood"] = "Quartiere futuro"
        validated = ames.validate_inference_input(probe)
        self.assertEqual(validated.iloc[0]["Neighborhood"], "Quartiere futuro")

    def test_inference_contract_accepts_python_and_numpy_numeric_scalars(self) -> None:
        probe = dict(self.valid_input)
        probe["Lot Area"] = np.float64(10000.0)
        probe["Overall Qual"] = np.int64(5)

        validated = ames.validate_inference_input(probe)

        self.assertEqual(float(validated.iloc[0]["Lot Area"]), 10000.0)
        self.assertEqual(float(validated.iloc[0]["Overall Qual"]), 5.0)

    def test_inference_contract_rejects_non_scalar_numeric_values(self) -> None:
        invalid_values = (
            np.array([100.0]),
            pd.Series([100.0]),
            [100.0],
            (100.0,),
            {"value": 100.0},
            {100.0},
            "10000",
        )
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                probe = dict(self.valid_input)
                probe["Lot Area"] = value
                with self.assertRaisesRegex(ValueError, "Lot Area"):
                    ames.validate_inference_input(probe)

    def test_inference_contract_rejects_boolean_numeric_values(self) -> None:
        for value in (True, np.bool_(True)):
            with self.subTest(value_type=type(value).__name__):
                probe = dict(self.valid_input)
                probe["Lot Area"] = value
                with self.assertRaisesRegex(ValueError, "booleano"):
                    ames.validate_inference_input(probe)

    def test_inference_contract_requires_integer_values_for_discrete_fields(self) -> None:
        invalid_updates = {
            "Overall Qual": {"Overall Qual": 5.5},
            "Overall Cond": {"Overall Cond": 5.5},
            "Mo Sold": {"Mo Sold": 6.5},
            "Year Built": {
                "Year Built": 2000.5,
                "Year Remod/Add": 2005,
                "Yr Sold": 2010,
            },
            "Year Remod/Add": {"Year Remod/Add": 2005.5, "Yr Sold": 2010},
            "Yr Sold": {"Yr Sold": 2010.5},
            "Garage Yr Blt": {"Garage Yr Blt": 2007.5, "Yr Sold": 2010},
        }
        for field, updates in invalid_updates.items():
            with self.subTest(field=field):
                probe = dict(self.valid_input)
                probe.update(updates)
                with self.assertRaisesRegex(ValueError, field):
                    ames.validate_inference_input(probe)

    def test_inference_contract_rejects_non_string_and_heterogeneous_extra_keys(self) -> None:
        invalid_key_sets = ((1,), (1, ("extra", 2)))
        for extra_keys in invalid_key_sets:
            with self.subTest(extra_keys=extra_keys):
                probe = dict(self.valid_input)
                for key in extra_keys:
                    probe[key] = "unexpected"
                with self.assertRaisesRegex(ValueError, "chiavi.*string"):
                    ames.validate_inference_input(probe)

    def test_inference_contract_rejects_impossible_years(self) -> None:
        probe = dict(self.valid_input)
        probe["Year Remod/Add"] = probe["Yr Sold"] + 1
        with self.assertRaisesRegex(ValueError, "Year Built <= Year Remod/Add <= Yr Sold"):
            ames.validate_inference_input(probe)

    def test_inference_contract_rejects_garage_contradiction(self) -> None:
        probe = dict(self.valid_input)
        probe["Garage Type"] = "None"
        probe["Garage Yr Blt"] = 2000
        probe["Garage Cars"] = 2
        probe["Garage Area"] = 400
        with self.assertRaisesRegex(ValueError, "Garage Yr Blt"):
            ames.validate_inference_input(probe)

    def test_predict_price_without_registered_model_raises_runtime_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Nessun modello finale registrato"):
            ames.predict_price(self.valid_input)

    def test_set_final_pipeline_rejects_unfitted_pipeline_without_registering_it(self) -> None:
        model = ames.build_model_pipeline(LinearRegression())

        with self.assertRaises(NotFittedError):
            ames.set_final_pipeline(model)

        self.assertIsNone(ames._FINAL_PIPELINE)

    def test_set_final_pipeline_rejects_pipeline_with_only_final_model_fitted(self) -> None:
        fitted_estimator = LinearRegression().fit([[0.0], [1.0]], [10.0, 20.0])
        partially_fitted_pipeline = ames.build_model_pipeline(fitted_estimator)
        ames._FINAL_PIPELINE = None

        with self.assertRaises(NotFittedError):
            ames.set_final_pipeline(partially_fitted_pipeline)

        self.assertIsNone(ames._FINAL_PIPELINE)

    def test_partial_fit_rejection_preserves_registered_ames_pipeline(self) -> None:
        valid_model = ames.build_model_pipeline(LinearRegression()).fit(
            self.X_development.iloc[:300], self.y_development.iloc[:300]
        )
        fitted_estimator = LinearRegression().fit([[0.0], [1.0]], [10.0, 20.0])
        partially_fitted_pipeline = ames.build_model_pipeline(fitted_estimator)
        ames.set_final_pipeline(valid_model)

        with self.assertRaises(NotFittedError):
            ames.set_final_pipeline(partially_fitted_pipeline)

        self.assertIs(ames._FINAL_PIPELINE, valid_model)

    def test_set_final_pipeline_rejects_fitted_estimator_outside_ames_pipeline(self) -> None:
        estimator = DummyRegressor().fit([[0.0], [1.0]], [10.0, 20.0])

        with self.assertRaisesRegex(TypeError, "Pipeline"):
            ames.set_final_pipeline(estimator)

        self.assertIsNone(ames._FINAL_PIPELINE)

    def test_set_final_pipeline_rejects_fitted_pipeline_with_different_steps(self) -> None:
        model = Pipeline([("regressor", DummyRegressor())]).fit(
            [[0.0], [1.0]], [10.0, 20.0]
        )

        with self.assertRaisesRegex(ValueError, "step"):
            ames.set_final_pipeline(model)

        self.assertIsNone(ames._FINAL_PIPELINE)

    def test_set_final_pipeline_rejects_wrong_ames_structural_component(self) -> None:
        model = ames.build_model_pipeline(LinearRegression()).fit(
            self.X_development.iloc[:300], self.y_development.iloc[:300]
        )
        model.steps[0] = ("wrangle", "passthrough")

        with self.assertRaisesRegex(ValueError, "wrangle"):
            ames.set_final_pipeline(model)

        self.assertIsNone(ames._FINAL_PIPELINE)

    def test_set_final_pipeline_rejects_unsupported_final_model(self) -> None:
        model = ames.build_model_pipeline(DummyRegressor()).fit(
            self.X_development.iloc[:300], self.y_development.iloc[:300]
        )

        with self.assertRaisesRegex(ValueError, "model"):
            ames.set_final_pipeline(model)

        self.assertIsNone(ames._FINAL_PIPELINE)

    def test_set_final_pipeline_rejection_preserves_registered_pipeline(self) -> None:
        valid_model = ames.build_model_pipeline(LinearRegression()).fit(
            self.X_development.iloc[:300], self.y_development.iloc[:300]
        )
        invalid_estimator = DummyRegressor().fit([[0.0], [1.0]], [10.0, 20.0])
        ames.set_final_pipeline(valid_model)

        with self.assertRaisesRegex(TypeError, "Pipeline"):
            ames.set_final_pipeline(invalid_estimator)

        self.assertIs(ames._FINAL_PIPELINE, valid_model)

    def test_linear_pipeline_and_predict_price_use_the_same_path(self) -> None:
        model = ames.build_model_pipeline(LinearRegression())
        model.fit(self.X_development.iloc[:300], self.y_development.iloc[:300])
        ames.set_final_pipeline(model)
        validated = ames.validate_inference_input(self.valid_input)
        expected = float(np.expm1(model.predict(validated)[0]))

        observed = ames.predict_price(self.valid_input)

        self.assertIsInstance(observed, float)
        self.assertTrue(np.isfinite(observed))
        self.assertGreater(observed, 0)
        self.assertAlmostEqual(observed, expected, places=10)

    def test_holdout_metrics_importance_and_ablation_on_development_only(self) -> None:
        training_X = self.X_development.iloc[:250]
        training_y = self.y_development.iloc[:250]
        local_validation_X = self.X_development.iloc[250:300]
        local_validation_y = self.y_development.iloc[250:300]

        model = ames.build_model_pipeline(LinearRegression()).fit(training_X, training_y)
        metrics, predictions, residuals = ames.evaluate_holdout(
            model,
            local_validation_X,
            local_validation_y,
        )
        self.assertEqual(metrics.shape, (1, 6))
        self.assertEqual(len(predictions), len(local_validation_X))
        self.assertEqual(len(residuals), len(local_validation_X))
        self.assertTrue(np.isfinite(metrics.to_numpy(dtype=float)).all())

        importance = ames.transformed_feature_importance(model)
        self.assertGreater(len(importance), len(ames.RAW_FEATURE_COLUMNS))
        self.assertTrue(np.isfinite(importance.to_numpy()).all())

        ablation_X = self.X_development.iloc[:200]
        ablation_y = self.y_development.iloc[:200]
        splits = ames.shared_cv_splits(ablation_X, ablation_y, n_splits=2)
        ablation = ames.feature_ablation_cv(model, ablation_X, ablation_y, splits)
        self.assertEqual(ablation.shape, (2, 3))
        self.assertTrue(np.isfinite(ablation.to_numpy()).all())

    @unittest.skipUnless(
        importlib.util.find_spec("xgboost") is not None,
        "XGBoost non è installato: impossibile verificare l'inventario completo dei modelli.",
    )
    def test_xgboost_search_inventory_can_be_built(self) -> None:
        splits = ames.shared_cv_splits(
            self.X_development.iloc[:100], self.y_development.iloc[:100]
        )
        searches = ames.build_tuned_searches(splits)
        self.assertEqual(set(searches), {"Ridge", "RandomForest", "XGBoost"})
        for search in searches.values():
            self.assertEqual(len(search.cv), len(splits))
            for observed, expected in zip(search.cv, splits, strict=True):
                np.testing.assert_array_equal(observed[0], expected[0])
                np.testing.assert_array_equal(observed[1], expected[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
