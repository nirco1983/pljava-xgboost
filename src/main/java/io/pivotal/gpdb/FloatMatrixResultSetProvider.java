package io.pivotal.gpdb;
import org.postgresql.pljava.ResultSetProvider;

import java.sql.ResultSet;
import java.sql.SQLException;

public class FloatMatrixResultSetProvider implements ResultSetProvider {
    private final float[][] matrix;

    public FloatMatrixResultSetProvider(float[][] matrix) {
        this.matrix = matrix;
    }

    private static Float[] toObjectArray(float[] array) {
        Float[] objArray = new Float[array.length];
        for (int i = 0; i < array.length; ++i) {
            objArray[i] = array[i];
        }
        return objArray;
    }

    @Override
    public boolean assignRowValues(ResultSet resultSet, int i) throws SQLException {
        if (i >= matrix.length)
            return false;
        resultSet.updateObject(1, toObjectArray(matrix[i]));
        return true;
    }

    @Override
    public void close() {
    }
}
