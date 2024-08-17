package org.flag4j.operations_old.dense.real;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseMatMultTransposeTests {
    double[][] entriesA, entriesB;
    MatrixOld A, B;
    double[] exp, act;

    @Test
    void squareTestCase() {
        entriesA = new double[][]{{1.1119431, 2.1, -3}, {1.33, -9.44, 1233.4}, {0.001, 0, -9.3}};
        entriesB = new double[][]{{10.3, 23, 4}, {1334.5, -13.4, 0}, {0.0013, 1, 34.5}};
        A = new MatrixOld(entriesA);
        B = new MatrixOld(entriesB);
        exp = A.mult(B.T()).entries;

        // ------------ Sub-case 1 ------------
        act = RealDenseMatrixMultTranspose.multTranspose(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = RealDenseMatrixMultTranspose.multTransposeBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = RealDenseMatrixMultTranspose.multTransposeConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = RealDenseMatrixMultTranspose.multTransposeBlockedConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new double[][]{
                {1.1119431, 2.1, -3},
                {1.33, -9.44, 1233.4},
                {0.001, 0, -9.3},
                {9.444, 13.4, -93.4}};
        entriesB = new double[][]{
                {10.3, 23, 4, 16.4, 7.8, 1.233},
                {1334.5, -13.4, 0, 11.3346, -0.3331334, 6.245},
                {0.0013, 1, 34.5, 4.566, -581.4, 0}};
        A = new MatrixOld(entriesA);
        B = new MatrixOld(entriesB).T();
        exp = A.mult(B.T()).entries;

        // ------------ Sub-case 1 ------------
        act = RealDenseMatrixMultTranspose.multTranspose(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = RealDenseMatrixMultTranspose.multTransposeBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = RealDenseMatrixMultTranspose.multTransposeConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = RealDenseMatrixMultTranspose.multTransposeBlockedConcurrent(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }
}
