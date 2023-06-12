package com.flag4j.operations.dense_sparse.real;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import org.junit.jupiter.api.Test;

import static com.flag4j.operations.dense_sparse.real.RealDenseSparseMatrixMultiplication.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseSparseMatMultTests {
    double[] bEntries;
    int[] rowIndices, colIndices;
    SparseMatrix B;
    Shape bShape;

    double[][] aEntries, expEntries;
    Matrix A, exp;

    @Test
    void matMultTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7375568794, 0.00143541},
                {-515.255376035, -10.7763114},
                {-0.00012148943299999999, 0.0},
                {-11.4330901794, -115804.09409999999}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.entries, standard(A.entries, A.shape, B.entries, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(exp.entries, concurrentStandard(A.entries, A.shape, B.entries, B.rowIndices, B.colIndices, B.shape));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(5, 3);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{0.0, 0.0, 0.0},
                {-1.04985560794, -92.7375568794, -0.00011494769430000002},
                {-10881.6915, 6434.2545, -10.7763114},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.entries, standard(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
        assertArrayEquals(exp.entries, concurrentStandard(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
    }

    @Test
    void matVecMultTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 0};
        bShape = new Shape(3, 1);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7361214694},
                {-526.0316874350001},
                {-0.00012148943299999999},
                {-115815.52719017939}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.entries, standardVector(A.entries, A.shape, B.entries, B.rowIndices));
        assertArrayEquals(exp.entries, concurrentStandardVector(A.entries, A.shape, B.entries, B.rowIndices));
        assertArrayEquals(exp.entries, blockedVector(A.entries, A.shape, B.entries, B.rowIndices));
        assertArrayEquals(exp.entries, concurrentBlockedVector(A.entries, A.shape, B.entries, B.rowIndices));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45},
                {123.445},
                {78.234}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(5, 4);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{0.0},
                {-1.04985560794},
                {-10881.6915},
                {0.0},
                {0.0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.entries, standardVector(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
        assertArrayEquals(exp.entries, concurrentStandardVector(B.entries, B.rowIndices, B.colIndices, B.shape, A.entries, A.shape));
    }
}
