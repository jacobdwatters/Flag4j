package com.flag4j.linalg.decompositions;

import com.flag4j.dense.Matrix;
import com.flag4j.linalg.decompositions.cholesky.RealCholeskyDecomposition;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealCholeskyTests {

    static double[][] aEntries;
    static double[][] expLEntries;

    static Matrix A;
    static Matrix expL;


    static void setMatrices() {
        A = new Matrix(aEntries);
        expL = new Matrix(expLEntries);
    }


    @Test
    void choleskyTestCase() {
        RealCholeskyDecomposition cholesky = new RealCholeskyDecomposition();

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{
                {2.0, -1.0, 0.0},
                {-1.0, 2.0, -1.0},
                {0.0, -1.0, 2.0}};
        expLEntries = new double[][]{
                {1.4142135623730951, 0.0, 0.0},
                {-0.7071067811865475, 1.224744871391589, 0.0},
                {0.0, -0.8164965809277261, 1.1547005383792515}};
        setMatrices();

        cholesky.decompose(A);
        assertEquals(expL, cholesky.getL());
    }
}
