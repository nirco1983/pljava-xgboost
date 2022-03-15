import io.pivotal.gpdb.XGBoostUdf;
import lombok.var;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.sql.DriverManager;
import java.util.*;

public class XGBoostTest {
    @Test
    public void glassTest() {
        var features = new ArrayList<Float>();
        var labels = new ArrayList<Integer>();
        try(var conn = DriverManager.getConnection("jdbc:postgresql://10.21.6.20/mydb?user=gpadmin")) {
            try (var stmt = conn.createStatement()) {
                var results = stmt.executeQuery("SELECT lbl,a,b FROM simple_data_rnd ORDER BY \"random\";");
                while (results.next()) {
                    var lbl = results.getInt("lbl");
                    var a = results.getFloat("a");
                    var b = results.getFloat("b");
                    features.add(a);
                    features.add(b);
                    labels.add(lbl);
                }
            }
        } catch (Exception ex) {
            Assertions.fail(ex);
        }
        var trainVector = new float[features.size() / 4];
        var trainLabels = new float[labels.size() / 4];
        var testVector = new float[features.size() * 3 / 4];
        var testLabels = new float[labels.size() * 3 / 4];
        int i;
        for (i = 0; i < trainLabels.length; ++i) {
            trainVector[2*i] = features.get(2*i);
            trainVector[2*i+1] = features.get(2*i+1);
            trainLabels[i] = labels.get(i);
        }
        for (int j = 0; i < labels.size(); j++, i++) {
            testVector[2*j] = features.get(2+i);
            testVector[2*j+1] = features.get(2*i+1);
            testLabels[j] = labels.get(i);
        }
        try {
            var trainMat = new DMatrix(trainVector, trainLabels.length, trainVector.length / trainLabels.length, Float.NaN);
            trainMat.setLabel(trainLabels);
            var testMat = new DMatrix(testVector, testLabels.length, testVector.length / testLabels.length, Float.NaN);
            testMat.setLabel(testLabels);
            Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
                {
                    put("train", trainMat);
                    put("test", testMat);
                }
            };
            var xgboostParams = new HashMap<String, Object>();
            xgboostParams.put("num_class", 3);
            var booster = XGBoost.train(trainMat, xgboostParams, 5, watches, null, null);
            var bytes = booster.toByteArray();
            booster = XGBoost.loadModel(bytes);
            var preditions = booster.predict(testMat);
            for (var j = 0; j < preditions.length; ++j) {
                System.out.println(testLabels[j] + ": " + Arrays.toString(preditions[j]));
            }
        }
        catch (Exception ex) {
            Assertions.fail(ex);
        }
    }

    private static float[] FloatArrayFromFloatList(List<Float> numbers) {
        var floats =new float[numbers.size()];
        for (int i = 0; i < floats.length; ++i)
            floats[i] = numbers.get(i);
        return floats;
    }

    private static float[] FloatArrayFromIntegerList(List<Integer> numbers) {
        var floats =new float[numbers.size()];
        for (int i = 0; i < floats.length; ++i)
            floats[i] = numbers.get(i);
        return floats;
    }

    @Test
    public void glassTest1() {
        var trainFeatureList = new ArrayList<Float>();
        var trainLabelList = new ArrayList<Integer>();
        var testFeatureList = new ArrayList<Float>();
        var testLabelList = new ArrayList<Integer>();
        byte[] dbModel = null;
        try(var conn = DriverManager.getConnection("jdbc:postgresql://10.21.6.20/mydb?user=gpadmin")) {
            try (var stmt1 = conn.createStatement()) {
                try (var stmt2 = conn.createStatement()) {
                    var train = stmt1.executeQuery("SELECT lbl,a,b FROM simple_data_rnd where \"random\" > 0.25;");
                    var test = stmt2.executeQuery("SELECT lbl,a,b FROM simple_data_rnd where \"random\" <= 0.25;");
                    while (train.next()) {
                        var lbl = train.getInt("lbl");
                        var a = train.getFloat("a");
                        var b = train.getFloat("b");
                        trainFeatureList.add(a);
                        trainFeatureList.add(b);
                        trainLabelList.add(lbl);
                    }
                    while (test.next()) {
                        var lbl = test.getInt("lbl");
                        var a = test.getFloat("a");
                        var b = test.getFloat("b");
                        testFeatureList.add(a);
                        testFeatureList.add(b);
                        testLabelList.add(lbl);
                    }
                }
            }
            try (var stmt = conn.createStatement()) {
                var result = stmt.executeQuery("SELECT model FROM simple_model;");
                result.next();
                dbModel = result.getBytes("model");
            }
        } catch (Exception ex) {
            Assertions.fail(ex);
        }
        var trainVector = FloatArrayFromFloatList(trainFeatureList);
        var testVector = FloatArrayFromFloatList(testFeatureList);
        var trainLabels = FloatArrayFromIntegerList(trainLabelList);
        var testLabels = FloatArrayFromIntegerList(testLabelList);
        try {
            var trainMat = new DMatrix(trainVector, trainLabels.length, trainVector.length / trainLabels.length, Float.NaN);
            trainMat.setLabel(trainLabels);
            var testMat = new DMatrix(testVector, testLabels.length, testVector.length / testLabels.length, Float.NaN);
            testMat.setLabel(testLabels);
            Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
                {
                    put("train", trainMat);
                    put("test", testMat);
                }
            };
            var xgboostParams = new HashMap<String, Object>();
            xgboostParams.put("num_class", 3);
            var booster = XGBoost.train(trainMat, xgboostParams, 5, watches, null, null);
            var bytes = booster.toByteArray();
            Assertions.assertArrayEquals(dbModel, bytes);
            booster = XGBoost.loadModel(bytes);
            var dbBooster = XGBoost.loadModel(dbModel);
            var preditions = booster.predict(testMat);
            var dbPreditions = dbBooster.predict(testMat);
            for (var j = 0; j < preditions.length; ++j) {
                System.out.println(testLabels[j] + ": " + Arrays.toString(preditions[j]));
                Assertions.assertArrayEquals(dbPreditions[j], preditions[j]);
            }
        }
        catch (Exception ex) {
            Assertions.fail(ex);
        }
    }

    @Test
    public void glassTest4() {
        var testFeatureList = new ArrayList<Float>();
        var testLabelList = new ArrayList<Integer>();
        byte[] dbModel = null;
        try(var conn = DriverManager.getConnection("jdbc:postgresql://10.21.6.20/mydb?user=gpadmin")) {
            try (var stmt = conn.createStatement()) {
                var test = stmt.executeQuery("SELECT lbl,a,b FROM simple_data_rnd where \"random\" <= 0.25;");
                while (test.next()) {
                    var lbl = test.getInt("lbl");
                    var a = test.getFloat("a");
                    var b = test.getFloat("b");
                    testFeatureList.add(a);
                    testFeatureList.add(b);
                    testLabelList.add(lbl);
                }
            }
            try (var stmt = conn.createStatement()) {
                var result = stmt.executeQuery("SELECT model FROM simple_model;");
                result.next();
                dbModel = result.getBytes("model");
            }
        } catch (Exception ex) {
            Assertions.fail(ex);
        }
        var testVector = FloatArrayFromFloatList(testFeatureList);
        var testLabels = FloatArrayFromIntegerList(testLabelList);
        try {
            var testMat = new DMatrix(testVector, testLabels.length, testVector.length / testLabels.length, Float.NaN);
            //testMat.setLabel(testLabels);
            //var xgboostParams = new HashMap<String, Object>();
            //xgboostParams.put("num_class", 3);
            //var booster = XGBoost.train(trainMat, xgboostParams, 5, watches, null, null);
            var dbBooster = XGBoost.loadModel(dbModel);
            var dbPreditions = dbBooster.predict(testMat);
            for (var j = 0; j < dbPreditions.length; ++j) {
                System.out.println(testLabels[j] + ": " + Arrays.toString(dbPreditions[j]));
            }
        }
        catch (Exception ex) {
            Assertions.fail(ex);
        }
    }

    @Test
    public void glassTest2() {
        try(var conn = DriverManager.getConnection("jdbc:postgresql://10.21.6.20/mydb?user=gpadmin")) {
            try (var stmt1 = conn.createStatement()) {
                try (var stmt2 = conn.createStatement()) {
                    var train = stmt1.executeQuery("SELECT array_agg(lbl),array_agg(a),array_agg(b) FROM simple_data_rnd where \"random\" > 0.25;");
                    var test = stmt2.executeQuery("SELECT array_agg(lbl),array_agg(a),array_agg(b) FROM simple_data_rnd where \"random\" <= 0.25;");
                    train.next();
                    test.next();
                    var modelBytes = XGBoostUdf.train(train, test, 5, "{\"num_class\": 3}");
                    var predictions = XGBoostUdf.predict(modelBytes, test, false, 0);
                    while (predictions.hasNext()) {
                        var prediction = (Float[]) predictions.next();
                        System.out.println(Arrays.toString(prediction));
                    }
                }
            }
        } catch (Exception ex) {
            Assertions.fail(ex);
        }
    }
}
