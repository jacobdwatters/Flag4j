package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.linalg.ops.DirectSum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixDirectSumTests {

    double[][] aEntries, bEntries, expEntries;
    CNumber[][] bComplexEntries, expComplexEntries;
    double[] bSparseEntries;
    CNumber[] bSparseComplexEntries;

    int[] rowIndices, colIndices;
    Shape sparseShape;

    MatrixOld A, B, exp;
    CMatrixOld BComplex, expComplex;
    CooMatrix BSparse;
    CooCMatrix BSparseComplex;

    @Test
    void matrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
        bEntries = new double[][]{{-9, 1.234}, {89.1, 0}, {0, -4.6}};
        B = new MatrixOld(bEntries);

        expEntries = new double[][]{
                {1, 2, 3, 0, 0},
                {4, 5, 6, 0, 0},
                {0, 0, 0, -9, 1.234},
                {0, 0, 0, 89.1, 0},
                {0, 0, 0, 0, -4.6}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void sparseMatrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
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
        exp = new MatrixOld(expEntries);

        assertEquals(exp, DirectSum.directSum(A, BSparse));
    }


    @Test
    void complexMatrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[][]{{new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)}};
        BComplex = new CMatrixOld(bComplexEntries);

        expComplexEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(0), new CNumber(0)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)}};
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, DirectSum.directSum(A, BComplex));
    }

    @Test
    void sparseComplexMatrixTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
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
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, DirectSum.directSum(A, BSparseComplex));
    }

    //--------------------------------------------------------------------------------------------------------------

    @Test
    void matrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
        bEntries = new double[][]{{-9, 1.234}, {89.1, 0}, {0, -4.6}};
        B = new MatrixOld(bEntries);
        expEntries = new double[][]{
                {0, 0, 0, -9, 1.234},
                {0, 0, 0, 89.1, 0},
                {0, 0, 0, 0, -4.6},
                {1, 2, 3, 0, 0},
                {4, 5, 6, 0, 0}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void sparseMatrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
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
        exp = new MatrixOld(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, BSparse));
    }


    @Test
    void complexMatrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[][]{{new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)}};
        BComplex = new CMatrixOld(bComplexEntries);

        expComplexEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0, 1), new CNumber(8.13)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1.44, -9.436), new CNumber(6.71, 8.44)},
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(0), new CNumber(0)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(0), new CNumber(0)}};
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, DirectSum.invDirectSum(A, BComplex));
    }

    @Test
    void sparseComplexMatrixInvTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        A = new MatrixOld(aEntries);
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
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, DirectSum.invDirectSum(A, BSparseComplex));
    }
}
