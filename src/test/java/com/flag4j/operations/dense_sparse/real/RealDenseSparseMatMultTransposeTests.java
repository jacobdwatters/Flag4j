package com.flag4j.operations.dense_sparse.real;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import org.junit.jupiter.api.Test;

import static com.flag4j.operations.dense_sparse.real.RealDenseSparseMatrixMultTranspose.multTranspose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseSparseMatMultTransposeTests {
    double[] bEntries;
    int[] rowIndices, colIndices;
    SparseMatrix B;
    Shape bShape;

    double[][] aEntries;
    Matrix A, exp;

    @Test
    void matMultTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 2};
        bShape = new Shape(2, 3);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);
        exp = A.mult(new SparseMatrix(bShape.copy().swapAxes(0, 1), bEntries, colIndices, rowIndices));

        assertArrayEquals(exp.entries, multTranspose(A.entries, A.shape, B.entries, B.rowIndices, B.colIndices, B.shape));
    }
}
