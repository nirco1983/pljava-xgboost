package io.pivotal.gpdb;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.var;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.postgresql.pljava.ResultSetProvider;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

@SuppressWarnings("unused")
public final class XGBoostUdf {
    final static Logger log = Logger.getAnonymousLogger();

    private static DMatrix toDMatrixWithLabels(
            ResultSet arraysRow,
            float missing) throws SQLException, XGBoostError {
        final var nColumns = arraysRow.getMetaData().getColumnCount();
        final var nFeatures = nColumns - 1;
        Number[][] featureArrays = null;
        if (nColumns == 1) {
            var dataObj = arraysRow.getObject(1);
            if (dataObj == null)
                return null;
            if (dataObj.getClass().getComponentType().isArray()) {
                // this is a matrix.
                featureArrays = (Number[][]) dataObj;
            }
        }
        if (featureArrays == null) {
            featureArrays = new Number[nColumns][];
            for (int col = 0; col < nColumns; ++col) {
                featureArrays[col] = (Number[]) arraysRow.getObject(col + 1);
            }
        }
        final int nRows = featureArrays[0].length;
        var features = new float[nRows * nFeatures];
        var labels = new float[nRows];
        for (int row = 0, i = 0; row < nRows; ++row) {
            // assumption: the first column is the label column.
            labels[row] = featureArrays[0][row].floatValue();
            for (int col = 1; col < nColumns; ++col, ++i) {
                features[i] = featureArrays[col][row].floatValue();
            }
        }
        var matrix = new DMatrix(features, nRows, nFeatures, missing);
        matrix.setLabel(labels);
        return matrix;
    }

    private static DMatrix toDMatrixWithoutLabels(
            ResultSet arraysRow,
            float missing) throws SQLException, XGBoostError {
        final var nColumns = arraysRow.getMetaData().getColumnCount();
        Number[][] featureArrays = null;
        if (nColumns == 1) {
            var dataObj = arraysRow.getObject(1);
            if (dataObj == null)
                return null;
            if (dataObj.getClass().getComponentType().isArray()) {
                // this is a matrix.
                featureArrays = (Number[][]) dataObj;
            }
        }
        if (featureArrays == null) {
            featureArrays = new Number[nColumns][];
            for (int col = 0; col < nColumns; ++col) {
                featureArrays[col] = (Number[]) arraysRow.getObject(col + 1);
            }
        }
        final int nRows = featureArrays[0].length;
        var features = new float[nRows * nColumns];
        for (int row = 0, i = 0; row < nRows; ++row) {
            for (int col = 0; col < nColumns; ++col, ++i) {
                features[i] = featureArrays[col][row].floatValue();
            }
        }
        return new DMatrix(features, nRows, nColumns, missing);
    }

    private static float[] toFloatArray(ResultSet arrayRow) throws SQLException {
        var numbers = (Number[])arrayRow.getObject(1);
        var array = new float[numbers.length];
        for (int i = 0; i < array.length; ++i) {
            array[i] = numbers[i].floatValue();
        }
        return array;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseXGBoostParams(String xgboostParamsJson) throws JsonProcessingException {
        Map<String, Object> xgboostParams;
        if (xgboostParamsJson != null) {
            //noinspection unchecked
            xgboostParams = new ObjectMapper().readValue(xgboostParamsJson, HashMap.class);
        } else {
            xgboostParams = new HashMap<>();
        }
        // disable multi-threading (not relevant in Greenplum):
        //noinspection SpellCheckingInspection
        xgboostParams.put("nthread", 1);
        return xgboostParams;
    }

    public static byte[] train(
            ResultSet trainData,
            ResultSet testData,
            float missing,
            int rounds,
            String xgboostParamsJson) {
        try {
            var xgboostParams = parseXGBoostParams(xgboostParamsJson);
            // build DMatrices:
            var trainMatrix = toDMatrixWithLabels(trainData, missing);
            var testMatrix = toDMatrixWithLabels(testData, missing);
            // Specify a watch list to see model accuracy on data sets
            Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
                {
                    put("train", trainMatrix);
                    put("test", testMatrix);
                }
            };
            var booster = XGBoost.train(trainMatrix, xgboostParams, rounds, watches, null, null);
            return booster.toByteArray();
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static byte[] train(
            ResultSet trainData,
            ResultSet testData,
            int rounds,
            String xgboostParamsJson) {
        return train(trainData, testData, Float.NaN, rounds, xgboostParamsJson);
    }

    private static Float[][] toObjectArray(float[][] array) {
        Float[][] objArray = new Float[array.length][];
        for (int row = 0; row < array.length; ++row) {
            var currentRow = array[row];
            var currentObjRow = objArray[row] = new Float[currentRow.length];
            for (int col = 0; col < currentObjRow.length; ++col) {
                currentObjRow[col] = currentRow[col];
            }
        }
        return objArray;
    }

    public static ResultSetProvider predict(
            byte[] modelBytes,
            ResultSet data,
            float missing,
            boolean outputMargin,
            int treeLimit,
            String xgboostParamsJson) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            // build DMatrices:
            var matrix = toDMatrixWithoutLabels(data, missing);
            var booster = XGBoost.loadModel(modelBytes);
            // set params if specified:
            if (xgboostParamsJson != null) {
                var xgboostParams = parseXGBoostParams(xgboostParamsJson);
                booster.setParams(xgboostParams);
            }
            var predictions = booster.predict(matrix, outputMargin, treeLimit);
            return new FloatMatrixResultSetProvider(predictions);
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static ResultSetProvider predict(
            byte[] modelBytes,
            ResultSet data,
            boolean outputMargin,
            int treeLimit,
            String xgboostParamsJson) {
        return predict(modelBytes, data, Float.NaN, outputMargin, treeLimit, xgboostParamsJson);
    }

    public static ResultSetProvider predict(
            byte[] modelBytes,
            ResultSet data,
            boolean outputMargin,
            int treeLimit) {
        return predict(modelBytes, data, Float.NaN, outputMargin, treeLimit, null);
    }

    public static ResultSetProvider predict(
            byte[] modelBytes,
            ResultSet data,
            boolean outputMargin) {
        return predict(modelBytes, data, Float.NaN, outputMargin, 0, null);
    }

    public static ResultSetProvider predict(
            byte[] modelBytes,
            ResultSet data) {
        return predict(modelBytes, data, Float.NaN, false, 0, null);
    }

    public static ResultSetProvider predict(
            byte[] modelBytes,
            ResultSet data,
            String xgboostParamsJson) {
        return predict(modelBytes, data, Float.NaN, false, 0, xgboostParamsJson);
    }

    public static ResultSetProvider predictLeaf(
            byte[] modelBytes,
            ResultSet data,
            float missing,
            int treeLimit,
            String xgboostParamsJson) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            // build DMatrices:
            var matrix = toDMatrixWithoutLabels(data, missing);
            var booster = XGBoost.loadModel(modelBytes);
            // set params if specified:
            if (xgboostParamsJson != null) {
                var xgboostParams = parseXGBoostParams(xgboostParamsJson);
                booster.setParams(xgboostParams);
            }
            var predictions = booster.predictLeaf(matrix, treeLimit);
            return new FloatMatrixResultSetProvider(predictions);
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static ResultSetProvider predictLeaf(
            byte[] modelBytes,
            ResultSet data,
            int treeLimit,
            String xgboostParamsJson) {
        return predictLeaf(modelBytes, data, Float.NaN, treeLimit, xgboostParamsJson);
    }

    public static ResultSetProvider predictLeaf(
            byte[] modelBytes,
            ResultSet data,
            int treeLimit) {
        return predictLeaf(modelBytes, data, Float.NaN, treeLimit, null);
    }

    public static ResultSetProvider predictLeaf(
            byte[] modelBytes,
            ResultSet data) {
        return predictLeaf(modelBytes, data, Float.NaN, 0, null);
    }

    public static ResultSetProvider predictContrib(
            byte[] modelBytes,
            ResultSet data,
            float missing,
            int treeLimit,
            String xgboostParamsJson) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            // build DMatrices:
            var matrix = toDMatrixWithoutLabels(data, missing);
            var booster = XGBoost.loadModel(modelBytes);
            // set params if specified:
            if (xgboostParamsJson != null) {
                var xgboostParams = parseXGBoostParams(xgboostParamsJson);
                booster.setParams(xgboostParams);
            }
            var predictions = booster.predictContrib(matrix, treeLimit);
            return new FloatMatrixResultSetProvider(predictions);
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static ResultSetProvider predictContrib(
            byte[] modelBytes,
            ResultSet data,
            int treeLimit,
            String xgboostParamsJson) {
        return predictContrib(modelBytes, data, Float.NaN, treeLimit, xgboostParamsJson);
    }

    public static ResultSetProvider predictContrib(
            byte[] modelBytes,
            ResultSet data,
            int treeLimit) {
        return predictContrib(modelBytes, data, Float.NaN, treeLimit, null);
    }

    public static ResultSetProvider predictContrib(
            byte[] modelBytes,
            ResultSet data) {
        return predictContrib(modelBytes, data, Float.NaN, 0, null);
    }

    public static byte[] update(
            byte[] modelBytes,
            ResultSet trainData,
            float missing,
            @SuppressWarnings("SpellCheckingInspection") int iter,
            String xgboostParamsJson) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            var trainMatrix = toDMatrixWithLabels(trainData, missing);
            // set params if specified:
            if (xgboostParamsJson != null) {
                var xgboostParams = parseXGBoostParams(xgboostParamsJson);
                booster.setParams(xgboostParams);
            }
            booster.update(trainMatrix, iter);
            return booster.toByteArray();
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static byte[] update(
            byte[] modelBytes,
            ResultSet trainData,
            @SuppressWarnings("SpellCheckingInspection") int iter,
            String xgboostParamsJson) {
        return update(modelBytes, trainData, Float.NaN, iter, xgboostParamsJson);
    }

    public static byte[] update(
            byte[] modelBytes,
            ResultSet trainData,
            @SuppressWarnings("SpellCheckingInspection") int iter) {
        return update(modelBytes, trainData, Float.NaN, iter, null);
    }

    public static byte[] update(
            byte[] modelBytes,
            ResultSet trainData,
            float missing,
            @SuppressWarnings("SpellCheckingInspection") int iter) {
        return update(modelBytes, trainData, Float.NaN, iter, null);
    }

    public static byte[] boost(
            byte[] modelBytes,
            ResultSet trainData,
            float missing,
            ResultSet gradRow,
            ResultSet hessRow,
            String xgboostParamsJson) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            // set params if specified:
            if (xgboostParamsJson != null) {
                var xgboostParams = parseXGBoostParams(xgboostParamsJson);
                booster.setParams(xgboostParams);
            }
            var trainMatrix = toDMatrixWithLabels(trainData, missing);
            var grad = toFloatArray(gradRow);
            var hess = toFloatArray(hessRow);
            booster.boost(trainMatrix, grad, hess);
            return booster.toByteArray();
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static byte[] boost(byte[] modelBytes,
                               ResultSet trainData,
                               ResultSet gradRow,
                               ResultSet hessRow,
                               String xgboostParamsJson) {
        return boost(modelBytes, trainData, Float.NaN, gradRow, hessRow, xgboostParamsJson);
    }

    public static byte[] boost(byte[] modelBytes,
                               ResultSet trainData,
                               ResultSet gradRow,
                               ResultSet hessRow) {
        return boost(modelBytes, trainData, Float.NaN, gradRow, hessRow, null);
    }

    public static byte[] boost(byte[] modelBytes,
                               ResultSet trainData,
                               float missing,
                               ResultSet gradRow,
                               ResultSet hessRow) {
        return boost(modelBytes, trainData, missing, gradRow, hessRow, null);
    }

    @SuppressWarnings("rawtypes")
    public static Iterator getModelDump(
            byte[] modelBytes,
            String featureNamesCsv,
            boolean withStats) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            String[] dump;
            var booster = XGBoost.loadModel(modelBytes);
            if (featureNamesCsv != null) {
                var featureNames = Arrays.stream(featureNamesCsv.split(",")).toArray(String[]::new);
                dump = booster.getModelDump(featureNames, withStats);
            } else {
                dump = booster.getModelDump((String) null, withStats);
            }
            return Arrays.stream(dump).iterator();
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    @SuppressWarnings("rawtypes")
    public static Iterator crossValidation(
            ResultSet trainData,
            float missing,
            int round,
            String xgboostParamsJson,
            @SuppressWarnings("SpellCheckingInspection") int nfold,
            String metricsCsv) {
        if (xgboostParamsJson == null)
            return null;
        try {
            var xgboostParams = parseXGBoostParams(xgboostParamsJson);
            // build DMatrices:
            var trainMatrix = toDMatrixWithLabels(trainData, missing);
            if (trainMatrix == null)
                return null;
            var metrics = metricsCsv != null ? metricsCsv.split(",") : null;
            var results = XGBoost.crossValidation(trainMatrix, xgboostParams, round, nfold, metrics, null, null);
            return Arrays.stream(results).iterator();
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    @SuppressWarnings("rawtypes")
    public static Iterator crossValidation(
            ResultSet trainData,
            int round,
            String xgboostParamsJson,
            @SuppressWarnings("SpellCheckingInspection") int nfold,
            String metricsCsv) {
        return crossValidation(trainData, Float.NaN, round, xgboostParamsJson, nfold, metricsCsv);
    }

    public static ResultSetProvider getFeatureScore(
            byte[] modelBytes,
            String featureNamesCsv) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            Map<String, Integer> featureScore;
            if (featureNamesCsv != null) {
                var featureNames = Arrays.stream(featureNamesCsv.split(",")).toArray(String[]::new);
                featureScore = booster.getFeatureScore(featureNames);
            } else {
                featureScore = booster.getFeatureScore((String) null);
            }
            var scoreIterator = featureScore.entrySet().iterator();
            return new ResultSetProvider() {
                @Override
                public boolean assignRowValues(ResultSet resultSet, int i) throws SQLException {
                    if (scoreIterator.hasNext()) {
                        var entry = scoreIterator.next();
                        resultSet.updateString(1, entry.getKey());
                        resultSet.updateInt(2, entry.getValue());
                        return true;
                    }
                    return false;
                }

                @Override
                public void close() {
                    // nothing to do.
                }
            };
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static ResultSetProvider getScore(
            byte[] modelBytes,
            String featureNamesCsv,
            String importanceType) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            Map<String, Double> score;
            if (featureNamesCsv != null) {
                var featureNames = Arrays.stream(featureNamesCsv.split(",")).toArray(String[]::new);
                score = booster.getScore(featureNames, importanceType);
            } else {
                score = booster.getScore((String) null, importanceType);
            }
            var scoreIterator = score.entrySet().iterator();
            return new ResultSetProvider() {
                @Override
                public boolean assignRowValues(ResultSet resultSet, int i) throws SQLException {
                    if (scoreIterator.hasNext()) {
                        var entry = scoreIterator.next();
                        resultSet.updateString(1, entry.getKey());
                        resultSet.updateDouble(2, entry.getValue());
                        return true;
                    }
                    return false;
                }

                @Override
                public void close() {
                    // nothing to do.
                }
            };
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static ResultSetProvider getAttrs(byte[] modelBytes) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            var scoreIterator =
                    booster.getAttrs().entrySet().iterator();
            return new ResultSetProvider() {
                @Override
                public boolean assignRowValues(ResultSet resultSet, int i) throws SQLException {
                    if (scoreIterator.hasNext()) {
                        var entry = scoreIterator.next();
                        resultSet.updateString(1, entry.getKey());
                        resultSet.updateString(2, entry.getValue());
                        return true;
                    }
                    return false;
                }

                @Override
                public void close() {
                    // nothing to do.
                }
            };
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static String getAttr(byte[] modelBytes, String key) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            return booster.getAttr(key);
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }

    public static Long getNumFeature(byte[] modelBytes) {
        try {
            if (modelBytes == null) {
                log.log(Level.WARNING, "model is NULL");
                return null;
            }
            var booster = XGBoost.loadModel(modelBytes);
            return booster.getNumFeature();
        } catch (Exception ex) {
            StringWriter sw = new StringWriter();
            PrintWriter st = new PrintWriter(sw);
            ex.printStackTrace(st);
            log.log(Level.WARNING, ex.getMessage() + "\n" + st);
            return null;
        }
    }
}