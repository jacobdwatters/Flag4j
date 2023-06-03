package com.flag4j.operations.sparse.real;

import com.flag4j.*;
import org.junit.jupiter.api.Test;

import static com.flag4j.operations.sparse.real.RealSparseMatrixMultiplication.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealSparseMatMultTests {
    double[][] expEntries;
    double[] aEntries, bEntries, bVectorEntries;

    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, indices;

    Shape aShape, bShape;

    SparseMatrix A, B;
    SparseVector bVector;
    Matrix exp;

    @Test
    void matMultTest() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new double[]{1, 9.43};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{2, 1};
        aShape = new Shape(4, 3);
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{1.34, 9234};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0, 9234},
                {0, 0},
                {0, 0},
                {0, 0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.entries, standard(A.entries, A.rowIndices, A.colIndices, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(exp.entries, concurrentStandard(A.entries, A.rowIndices, A.colIndices, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape));
    }


    @Test
    void matVecMultTest() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new double[]{1, 7.9, 9.43};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 0, 1};
        aShape = new Shape(4, 3);
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{1.34};
        indices = new int[]{1};
        bVector = new SparseVector(3, bEntries, indices);

        expEntries = new double[][]{{0}, {0}, {12.6362}, {0}};
        exp = new Matrix(expEntries);

        assertArrayEquals(exp.entries, standardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices, bVector.shape));
        assertArrayEquals(exp.entries, concurrentStandardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices, bVector.shape));
    }
}








