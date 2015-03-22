# kaggle-axa-driver-telematics
Code to generate a submission for the Axa driver telematics analysis competition hosted on Kaggle

Features were generated using an R script posted by jvorwald on the competition
forum at https://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/12299/any-suggestions-on-doing-local-evaluations/63084#post63084 with minimal modification.

Predictions were then generated using pandas and scikit-learn. I looked into
LogisticRegression and ExtraTressClassifier but RandomForestClassifier produced
the best result.

## Wait, that makes no sense . . .
The code could be simplified significantly, the remnants of untried and failed
experiments are still present.

### Exclude the driver of interest when picking negative examples
get_random_trip_collection was written with the ability to exclude a driver.
I never tried this and instead chose completely at random. The code was not
rewritten and instead 1000000 is passed which is never present.

### Scaling predictions
We aren't told how many negative examples are in each folder. I tried to scale
predictions to normalize across drivers but this didn't appear to work. The raw
predictions produced a slightly better result (0.88797 vs 0.88764).
