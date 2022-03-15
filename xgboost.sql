drop schema if exists xgboost cascade;

create schema xgboost;
grant usage on schema xgboost to public;

CREATE OR REPLACE FUNCTION xgboost.train(
	train_data record,
	test_data record,
	missing REAL,
	rounds INT,
	xgboostParamsJson JSON) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.train'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.train(
	train_data record,
	test_data record,
	missing REAL,
	rounds INT,
	xgboostParamsJson JSON) to public;

CREATE OR REPLACE FUNCTION xgboost.train(
	train_data record,
	test_data record,
	rounds INT,
	xgboostParamsJson JSON) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.train'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.train(
	train_data record,
	test_data record,
	rounds INT,
	xgboostParamsJson JSON) to public;

CREATE OR REPLACE FUNCTION xgboost.crossValidation(
	train_data record,
	missing REAL,
	rounds INT,
	xgboostParamsJson JSON,
	nfold INT,
	metricsCsv text) returns setof text as 'io.pivotal.gpdb.XGBoostUdf.crossValidation'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.crossValidation(
	train_data record,
	missing REAL,
	rounds INT,
	xgboostParamsJson JSON,
	nfold INT,
	metricsCsv text) to public;

CREATE OR replace FUNCTION xgboost.crossValidation(
	train_data record,
	rounds INT,
	xgboostParamsJson JSON,
	nfold INT,
	metricsCsv text) returns setof text as 'io.pivotal.gpdb.XGBoostUdf.crossValidation'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.crossValidation(
	train_data record,
	rounds INT,
	xgboostParamsJson JSON,
	nfold INT,
	metricsCsv text) to public;

create type xgboost.predict_t as (prediction real[]);
CREATE OR replace FUNCTION xgboost.predict(
	model bytea,
	features record,
	missing real,
	outputMargin boolean,
	treeLimit int,
	xgboostParamsJson JSON) returns setof xgboost.predict_t as 'io.pivotal.gpdb.XGBoostUdf.predict'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predict(
	model bytea,
	features record,
	missing real,
	outputMargin boolean,
	treeLimit int,
	xgboostParamsJson JSON) to public;

CREATE OR replace FUNCTION xgboost.predict(
	model bytea,
	features record,
	outputMargin boolean,
	treeLimit int,
	xgboostParamsJson JSON) returns setof xgboost.predict_t as 'io.pivotal.gpdb.XGBoostUdf.predict'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predict(
	model bytea,
	features record,
	outputMargin boolean,
	treeLimit int,
	xgboostParamsJson JSON) to public;

CREATE OR replace FUNCTION xgboost.predict(
	model bytea,
	features record,
	outputMargin boolean,
	treeLimit int) returns setof xgboost.predict_t as 'io.pivotal.gpdb.XGBoostUdf.predict'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predict(
	model bytea,
	features record,
	outputMargin boolean,
	treeLimit int) to public;

CREATE OR replace FUNCTION xgboost.predict(
	model bytea,
	features record,
	outputMargin boolean) returns setof xgboost.predict_t as 'io.pivotal.gpdb.XGBoostUdf.predict'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predict(
	model bytea,
	features record,
	outputMargin boolean) to public;

CREATE OR replace FUNCTION xgboost.predict(
	model bytea,
	features record,
	xgboostParamsJson json) returns setof xgboost.predict_t as 'io.pivotal.gpdb.XGBoostUdf.predict'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predict(
	model bytea,
	features record,
	xgboostParamsJson json) to public;


CREATE OR replace FUNCTION xgboost.predict(
	model bytea,
	features record) returns setof xgboost.predict_t as 'io.pivotal.gpdb.XGBoostUdf.predict'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predict(
	model bytea,
	features record) to public;

create type xgboost.predictLeaf_t as (prediction real[]);
CREATE OR replace FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	treeLimit int) returns setof xgboost.predictLeaf_t as 'io.pivotal.gpdb.XGBoostUdf.predictLeaf'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	treeLimit int) to public;

CREATE OR replace FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	missing real,
	treeLimit int) returns setof xgboost.predictLeaf_t as 'io.pivotal.gpdb.XGBoostUdf.predictLeaf'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	missing real,
	treeLimit int) to public;

CREATE OR replace FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	missing real,
	treeLimit int,
	xgboostParamsJson json) returns setof xgboost.predictLeaf_t as 'io.pivotal.gpdb.XGBoostUdf.predictLeaf'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	missing real,
	treeLimit int,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	treeLimit int,
	xgboostParamsJson json) returns setof xgboost.predictLeaf_t as 'io.pivotal.gpdb.XGBoostUdf.predictLeaf'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictLeaf(
	model bytea,
	features record,
	treeLimit int,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.predictLeaf(
	model bytea,
	features record) returns setof xgboost.predictLeaf_t as 'io.pivotal.gpdb.XGBoostUdf.predictLeaf'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictLeaf(
	model bytea,
	features record) to public;

create type xgboost.predictContrib_t as (prediction real[]);
CREATE OR replace FUNCTION xgboost.predictContrib(
	model bytea,
	features record,
	missing real,
	treeLimit int,
	xgboostParamsJson json) returns setof xgboost.predictContrib_t as 'io.pivotal.gpdb.XGBoostUdf.predictContrib'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictContrib(
	model bytea,
	features record,
	missing real,
	treeLimit int,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.predictContrib(
	model bytea,
	features record,
	treeLimit int,
	xgboostParamsJson json) returns setof xgboost.predictContrib_t as 'io.pivotal.gpdb.XGBoostUdf.predictContrib'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictContrib(
	model bytea,
	features record,
	treeLimit int,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.predictContrib(
	model bytea,
	features record,
	treeLimit int) returns setof xgboost.predictContrib_t as 'io.pivotal.gpdb.XGBoostUdf.predictContrib'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictContrib(
	model bytea,
	features record,
	treeLimit int) to public;

CREATE OR replace FUNCTION xgboost.predictContrib(
	model bytea,
	features record) returns setof xgboost.predictContrib_t as 'io.pivotal.gpdb.XGBoostUdf.predictContrib'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.predictContrib(
	model bytea,
	features record) to public;

CREATE OR replace FUNCTION xgboost.update(
	model bytea,
	trainData record,
	missing real,
	iter int,
	xgboostParamsJson json) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.update'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.update(
	model bytea,
	trainData record,
	missing real,
	iter int,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.update(
	model bytea,
	trainData record,
	iter int,
	xgboostParamsJson json) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.update'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.update(
	model bytea,
	trainData record,
	iter int,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.update(
	model bytea,
	trainData record,
	iter int) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.update'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.update(
	model bytea,
	trainData record,
	iter int) to public;

CREATE OR replace FUNCTION xgboost.update(
	model bytea,
	trainData record,
	missing real,
	iter int) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.update'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.update(
	model bytea,
	trainData record,
	missing real,
	iter int) to public;

CREATE OR replace FUNCTION xgboost.boost(
	model bytea,
	trainData record,
	missing real,
	grad record,
	hess record,
	xgboostParamsJson json) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.boost'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.boost(
	model bytea,
	trainData record,
	missing real,
	grad record,
	hess record,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.boost(
	model bytea,
	features record,
	grad record,
	hess record) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.boost'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.boost(
	model bytea,
	features record,
	grad record,
	hess record) to public;

CREATE OR replace FUNCTION xgboost.boost(
	model bytea,
	features record,
	grad record,
	hess record,
	xgboostParamsJson json) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.boost'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.boost(
	model bytea,
	features record,
	grad record,
	hess record,
	xgboostParamsJson json) to public;

CREATE OR replace FUNCTION xgboost.boost(
	model bytea,
	features record,
	missing real,
	grad record,
	hess record) returns bytea as 'io.pivotal.gpdb.XGBoostUdf.boost'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.boost(
	model bytea,
	features record,
	missing real,
	grad record,
	hess record) to public;

CREATE OR replace FUNCTION xgboost.getModelDump(
	model bytea,
	featureNamesCsv text,
	withStats boolean) returns setof text as 'io.pivotal.gpdb.XGBoostUdf.getModelDump'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.getModelDump(
	model bytea,
	featureNamesCsv text,
	withStats boolean) to public;

create type xgboost.getFeatureScore_t as (key text, score int);
CREATE OR replace FUNCTION xgboost.getFeatureScore(
	model bytea,
	featureNamesCsv text) returns setof xgboost.getFeatureScore_t as 'io.pivotal.gpdb.XGBoostUdf.getFeatureScore'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.getFeatureScore(
	model bytea,
	featureNamesCsv text) to public;

create type xgboost.getScore_t as (key text, score float);
create type xgboost.FeatureImportanceType as enum('weight', 'gain', 'cover', 'total_gain', 'total_cover');
CREATE OR replace FUNCTION xgboost.getScore(
	model bytea,
	featureNamesCsv text,
	importancetype xgboost.FeatureImportanceType) returns setof xgboost.getScore_t as 'io.pivotal.gpdb.XGBoostUdf.getScore'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.getScore(
	model bytea,
	featureNamesCsv text,
	importancetype xgboost.FeatureImportanceType) to public;


create type xgboost.getAttrs_t as(key text, value text);
CREATE OR replace FUNCTION xgboost.getAttrs(model bytea) returns setof xgboost.getAttrs_t as 'io.pivotal.gpdb.XGBoostUdf.getAttrs'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.getAttrs(model bytea) to public;

CREATE OR replace FUNCTION xgboost.getAttr(model bytea, key text) returns text as 'io.pivotal.gpdb.XGBoostUdf.getAttr'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.getAttr(model bytea, key text) to public;

CREATE OR replace FUNCTION xgboost.getNumFeature(model bytea) returns bigint as 'io.pivotal.gpdb.XGBoostUdf.getNumFeature'
	IMMUTABLE LANGUAGE javau;
grant execute on FUNCTION xgboost.getNumFeature(model bytea) to public;