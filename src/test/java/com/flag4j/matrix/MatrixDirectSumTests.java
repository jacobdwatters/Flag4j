package com.flag4j.matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixDirectSumTests {

    double[][] aEntries, bEntries, expEntries;
    CNumber[][] bComplexEntries, expComplexEntries;
    double[] bSparseEntries;
    CNumber[] bSparseComplexEntries;

    int[] rowIndices, colIndices;
    Shape sparseShape;

    Matrix A, B, exp;
    CMatrix BComplex, expComplex;
    CooMatrix BSparse;
    CooCMatrix BSparseComplex;

    @Test
    void matrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{-9, 1.234}, {89.1, 0}, {0, -4.6}};
        B = new Matrix(bEntries);

        expEntries = new double[][]{
                {1, 2, 3, 0, 0},
                {4, 5, 6, 0, 0},
                {0, 0, 0, -9, 1.234},
                {0, 0, 0, 89.1, 0},
                {0, 0, 0, 0, -4.6}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.directSum(B));
    }


    @Test
    void sparseMatrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseEntries = new double[]{9.32, 13.4, -80.1, 1001.0004};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparse = new CooMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        expEntries = new double[][]{
                {1, 2, 3, 0, 0, 0},
                {4, 5, 6, 0, 0, 0},
                {0, 0, 0, 9.32, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 13.4, -80.1, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1001.0004, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.directSum(BSparse));
    }


    @Test
    void complexMatrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bComplexEntries = new CNumber[][]{{new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)}};
        BComplex = new CMatrix(bComplexEntries);

        expComplexEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(0), new CNumber(0)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, A.directSum(BComplex));
    }

    @Test
    void sparseComplexMatrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseComplexEntries = new CNumber[]{new CNumber(0, 1), new CNumber(8.13),
                new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparseComplex = new CooCMatrix(sparseShape, bSparseComplexEntries, rowIndices, colIndices);

        expComplexEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(8.13), new CNumber(1.44, -9.436), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(6.71, 8.44), new CNumber(0)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, A.directSum(BSparseComplex));
    }

    //--------------------------------------------------------------------------------------------------------------

    @Test
    void matrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{-9, 1.234}, {89.1, 0}, {0, -4.6}};
        B = new Matrix(bEntries);
        expEntries = new double[][]{
                {0, 0, 0, -9, 1.234},
                {0, 0, 0, 89.1, 0},
                {0, 0, 0, 0, -4.6},
                {1, 2, 3, 0, 0},
                {4, 5, 6, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.invDirectSum(B));
    }


    @Test
    void sparseMatrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseEntries = new double[]{9.32, 13.4, -80.1, 1001.0004};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparse = new CooMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        expEntries = new double[][]{
                {0, 0, 0, 9.32, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 13.4, -80.1, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1001.0004, 0},
                {1, 2, 3, 0, 0, 0},
                {4, 5, 6, 0, 0, 0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.invDirectSum(BSparse));
    }


    @Test
    void complexMatrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bComplexEntries = new CNumber[][]{{new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)}};
        BComplex = new CMatrix(bComplexEntries);

        expComplexEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)},
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(0), new CNumber(0)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(0), new CNumber(0)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, A.invDirectSum(BComplex));
    }

    @Test
    void sparseComplexMatrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new Matrix(aEntries);
        bSparseComplexEntries = new CNumber[]{new CNumber(0, 1), new CNumber(8.13),
                new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)};
        rowIndices = new int[]{0, 2, 2, 4};
        colIndices = new int[]{0, 0, 1, 1};
        sparseShape = new Shape(5, 3);
        BSparseComplex = new CooCMatrix(sparseShape, bSparseComplexEntries, rowIndices, colIndices);

        expComplexEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(8.13), new CNumber(1.44, -9.436), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(6.71, 8.44), new CNumber(0)},
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(0), new CNumber(0), new CNumber(0)}};
        expComplex = new CMatrix(expComplexEntries);

        assertEquals(expComplex, A.invDirectSum(BSparseComplex));
    }
}
