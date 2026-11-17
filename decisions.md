# Decisions

This file is the history of the project. Append an entry when a choice changes the work: the quantity, the data, the split, the model, the loss, or the action. Leave old entries in place. A later entry can reverse an earlier one. If a result depends on a random draw, record the seed under Why. The split entry records the cutoff date, or the seed and the fraction. The pull request that merges the code names this entry.

## 2026-10-28: Data Choice

- Decision: Chose to use "A Reanalysis of the Original Test Data for the Taylor Standard Series" for the data. Other series may be added in the future to improve the model breadth as time allows.
- Why: This report is freely available with no redistribution restrictions. The alternative series is in a more usable form, but was unavailable at the time of attempted access.
- Alternative: Delft Systematic Yacht Hull Series
- Revisit: If the source website [data.4tu.nl](https://data.4tu.nl/articles/dataset/Delft_Systematic_Yacht_Hull_Series_Measurement_Data/21501402) comes back online in a reasonable time-frame, it may be used instead or as well.

## 2026-11-16: Data Reevaluation

- Decision: Chose to switch from Taylor Series to Delft Systematic Deadrise Series
- Why: The dataset is now available, and the data extraction from the Taylor Series report was inconsistent.
- Alternative: Manual data extraction from Taylor Series Report
- Revisit: No.

## 2026-11-16: Split Model Evaluation

- Decision: Perform separate evaluations of flow and hull parameters to determine effective way of representing the flow parameters before flow and hull parameters are combined.
- Why: Large parameter variations would be computational prohibitive.
- Alternative: Restrict the possible representations and variations of parameters (e.g. don't include logarithmic, inverse, or cubic variations on parameters.
- Revisit: If a better physics-based flow parameter representation is found.

## 2026-11-17: Model Product and Decision Units

- Decision: Treat total resistance coefficient C_T as the model product. Report primary prediction errors and asymmetric loss in dimensionless C_T coefficient units, not Newtons or percentage error.
- Why: This is the quantity directly modeled in the preliminary study. It avoids claiming an end-to-end resistance-force predictor when running wetted area is not available as a design input.
- Alternative: Predict resistance force, or a coefficient normalized using a design-known reference area.
- Revisit: If the project objective changes to dimensional resistance prediction. The current C_T uses measured running wetted area and cannot independently provide design-stage force predictions without that area.

## 2026-11-17: Preliminary Evaluation Split

- Decision: Use one GroupShuffleSplit with 20% of configuration–model–year groups held out and random_state=42. Save row-level assignments. Do not train on held-out rows. The fraction applies to groups, not necessarily rows.
- Why: Runs within an identifier group are related and must remain on the same side of the split. A single explicit comparison makes the preliminary baseline and model scores reproducible.
- Alternative: Random row splitting; five-fold grouped cross-validation; grouping by verified physical hull identity.
- Revisit: When physical-hull identifiers become available or the final evaluation is designed. Prior flow selection used the existing dataset, so this split is not historically untouched final validation. Stop further tuning against this held-out result; use grouped development validation for the hull search.

## 2026-11-17: Preliminary Baseline

- Decision: Predict a constant C_T equal to the training-only group-weighted 2/3 quantile. Give each training group equal total weight. Score it on the same held-out rows as the model.
- Why: This constant minimizes the stated asymmetric absolute loss and provides a simple, loss-aligned benchmark. It differs from the raw-Fn/raw-Re representation reference used in the flow study.
- Alternative: Training mean, median, or a simple regression.
- Revisit: If the loss ratio or prediction target changes.

## 2026-11-17: Preliminary Model and Inputs

- Decision: Use fixed histogram gradient boosting with squared-error training, 300 iterations, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0, early_stopping=False, and random_state=42. Inputs are Beta, Lp/Bpx, LCG, Ap/Nabla^(2/3), Ta, Theta_0, CAP, Nabla_Lp3, and sqrt(Fn). Exclude Re and measured running Trim for this preliminary candidate.
- Why: Square-root Fn without Re was a simple near-leading configuration in the prior flow study. This fixes one reproducible candidate without repeating the 288-candidate search. Re omission does not establish that viscosity or temperature is irrelevant.
- Alternative: Log-quadratic Fn with raw-plus-log Re; the log-cubic-Fn Ridge screening model; raw-flow representations.
- Revisit: After controlled hull/static-condition comparisons and the small combined flow–hull check, using development validation rather than tuning against this preliminary held-out score.

## 2026-11-17: Evaluation Loss

- Decision: Rank resistance-coefficient predictions using group-balanced asymmetric absolute error. Each underprediction incurs 2 times its absolute error; each overprediction incurs 1 times its absolute error. Average within groups, then equally across groups. Also report group-balanced MAE, pooled RMSE, pooled bias, and underprediction frequency.
- Why: Underprediction is assigned greater importance, while group balancing prevents heavily sampled groups from dominating the primary score. The factor of two is a provisional preference, not an empirically established engineering cost ratio.
- Alternative: Symmetric MAE or RMSE; a different asymmetric penalty; asymmetric model training.
- Revisit: If a justified cost ratio becomes available. Current model training remains symmetric squared error, so asymmetric scoring does not make predictions guaranteed conservative bounds.

## 2026-11-17: Uncertainty and Scope

- Decision: Report a paired bootstrap interval for the difference in held-out group losses, resampling groups 2,000 times with seed 42. Label it as conditional evaluation variability, not an individual prediction interval or independent final validation.
- Why: Baseline and model are evaluated on the same groups. Resampling those groups preserves the comparison and avoids treating all runs as independent.
- Alternative: No interval; nested grouped evaluation; external validation; separately calibrated predictive intervals.
- Revisit: Before final generalization claims. This interval omits feature-selection and training-sample uncertainty and can be unstable with few held-out groups.

