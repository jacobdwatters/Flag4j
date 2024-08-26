package org.flag4j.operations_old.dense_sparse.real;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.flag4j.operations_old.dense_sparse.coo.real.RealDenseSparseMatrixMultTranspose.multTranspose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealDenseSparseMatMultTransposeTests {
    double[] bEntries;
    int[] rowIndices, colIndices;
    CooMatrixOld B;
    Shape bShape;

    double[][] aEntries;
    MatrixOld A, exp;

    @Test
    void matMultTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 2};
        bShape = new Shape(2, 3);
        B = new CooMatrixOld(bShape, bEntries, rowIndices, colIndices);
        exp = A.mult(new CooMatrixOld(bShape.swapAxes(0, 1), bEntries, colIndices, rowIndices));

        assertArrayEquals(exp.entries, multTranspose(A.entries, A.shape, B.entries, B.rowIndices, B.colIndices, B.shape));
    }
}