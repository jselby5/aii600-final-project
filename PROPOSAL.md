# Proposal

A simple ML regression model for ship hull resistance.

## Hypothesis

An ML regression model using hull-form parameters and operating speed will estimate calm-water residual-resistance coefficient for the Taylor Standard Series more accurately than a regression baseline using speed alone when evaluated on held-out hull forms. The single target will be the residual-resistance coefficient, \(C_R\), which is dimensionless; estimates will apply to steady towing conditions at a specified Froude number.

## Data

The primary source is Morton Gertler’s *A Reanalysis of the Original Test Data for the Taylor Standard Series*, David Taylor Model Basin Report 806, published in March 1954. One observation will be a hull configuration at one Froude number, paired with its residual-resistance coefficient. The **sample size** will be determined after extracting usable points from the published curves. The original experiment was performed in 1943- the reanalysis was performed from 1948-1951 to align with updated standards (i.e. 1947 International Towing Tank Convention Standards). The data originate from towing-tank experiments, but the published curves are smoothed representations of the experimental results- post-hoc estimates for ambient temperature were used for the post-1947 viscosity calculation (originally temperature was not recorded).

The model will estimate a dimensionless residual-resistance coefficient, whereas a design decision normally requires full-scale total resistance and propulsion power (the coefficient is an effective proxy in early stages).

The original government-authored report was released for public distribution; the report and extracted data will be republished on GitHub. Other compatible test series may be added as time allows.

## Decision affected by the model

A naval architect conducting preliminary design would use predicted residual-resistance coefficients to compare hull alternatives within the Taylor Standard Series parameter range and prioritize candidates for further analysis or testing. For this screening decision, two arbitrary hull forms from other reports will be selected for comparison. **Underprediction** will be treated as the more costly mistake because it could make a higher-resistance hull appear favorable and understate its propulsion requirements, with cascading negative impacts on the overall system design.
