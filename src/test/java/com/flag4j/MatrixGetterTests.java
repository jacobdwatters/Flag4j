package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixGetterTests {

    Matrix A;
    int rowNum;
    double[][] aEntries;
    Double[] exp, act;

    @Test
    void getRowTest() {
        aEntries = new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        A = new Matrix(aEntries);
        int rowNum = 1;
        exp = new Double[]{5.0, 6.0, 7.0, 8.0};
        act = A.getRow(rowNum);
        assertArrayEquals(exp, act);
    }
}
