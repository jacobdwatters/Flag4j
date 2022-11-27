package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixAbsTest {

    Matrix A, act, exp;
    double[][] aEntries, expEntries;


    @Test
    void absTest() {
        aEntries = new double[][]
                {{1, 2, -4, -9910.43},
                {1, -0.224, 2, Double.NEGATIVE_INFINITY},
                {Double.NaN, 8003.33, -93.441, 2.94}};
        expEntries = new double[][]
                {{1, 2, 4, 9910.43},
                {1, 0.224, 2, Double.POSITIVE_INFINITY},
                {Double.NaN, 8003.33, 93.441, 2.94}};
        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);
        act = A.abs();
        assertArrayEquals(exp.entries, act.entries);
    }
}
