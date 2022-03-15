# pljava-xgboost
A PL/Java wrapper for XGBoost

Prerequisites:
- OpenBLAS-0.3.20 or later

Installation:
1. Add the jar file to the PL/Java classpath.
For Example:
```sh
cp pljava-xgboost-1.0-SNAPSHOT.jar $GPHOME/lib/postgresql/java/.
gpconfig -c pljava_classpath -v 'pljava-xgboost-1.0-SNAPSHOT.jar'
```
2. Run the xgboost.sql script.

Troubleshooting:
Set the log_min_messages to DEBUG in your session before executing any of the xgboost functions:
```sql
SET log_min_messages='debug';
```
It will expose much more information in the log that can be used to find the root-cause.

Example SQL:
```sql
-- In this example we assume that our dataset has two features (a,b) and a label (lbl):
CREATE TABLE simple_data
(
	lbl INT,
	a REAL,
	b REAL
);
-- Create a table for holding xgboost models:
CREATE TABLE simple_model
(
	id SERIAL NOT NULL PRIMARY KEY,
	model BYTEA NOT NULL
);
-- Train a model using softmax and mloglog:
-- ASSUMPTION: the labels are integers ranging from 0 to 2.
-- 1. Split our dataset randomly into training and test:
(LIKE simple_data);
CREATE TABLE simple_data_test(LIKE simple_data);
CREATE TEMP TABLE rnd_split AS
SELECT *, random() rnd
FROM simple_data;
CREATE TABLE simple_data_train AS
SELECT *
FROM rnd_split
WHERE rnd > 0.25;
CREATE TABLE simple_data_test AS
SELECT *
FROM rnd_split
WHERE rnd <= 0.25;
DROP TABLE rnd_split;
-- 2. Train a model:
INSERT INTO simple_model(model)
-- The dataset is provided in a single row containing arrays of numbers.
-- The array in the first column holds the labels;
-- The arrays in the rest of the columns hold the features.
-- The row should contain one column per feature.
WITH train(train_data) AS (
	SELECT
		row(
		array_agg(lbl), -- label array
		array_agg(a),   -- feature array for a
		array_agg(b))	-- feature array for b
	from simple_data_train
), test(test_data) AS (
	SELECT
		row(
		array_agg(lbl), -- label array
		array_agg(a),   -- feature array for a
		array_agg(b))	-- feature array for b
	FROM simple_data_test 
)
SELECT xgboost.train(train_data, test_data, 5 /* round */, json '{"num_class":3}')
FROM train
CROSS JOIN test;

-- Explore our model:
SELECT id, xgboost.getModelDump(model, 'a,b', true)
FROM simple_model;

SELECT key, value
FROM simple_model, xgboost.getAttrs(model);

-- Make predictions:
WITH dataset(data, labels) AS (
	SELECT 
		row(
		array_agg(a),   -- feature array for a
		array_agg(b)),	-- feature array for b
		array_agg(lbl) labels
	FROM simple_data_test
), predictions AS (
	SELECT
		-- the predict functions returns a row containing a single column of type real[] called "prediction".
		-- each element in the array is a prediction for the corresponding datapoint in the feature arrays.
		(xgboost.predict(model, data, false /* outputMargin */, 0 /* treeLimit */)).prediction[0] predicted,
		unnest(labels) lbl
	FROM models, dataset
)
SELECT AVG(POWER(SIGN(predicted - lbl), 2)) mse
FROM predictions;
```