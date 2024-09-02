package org.flag4j.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixStackTests {
    Shape sparseShape;
    int[] rowIndices, colIndices;
    CNumber[][] aEntries, expEntries;
    CMatrixOld A, exp;

    @Test
    void realMatrixTestCase() {
        double[][] bEntries;
        MatrixOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new MatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(13.45), new CNumber(5.5)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-94.3345), new CNumber(435.6)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{13.45, 5.5, 23.45}, {-94.3345, 435.6, -8234.2}, {3.67, -798.41, 45.6}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{13.45, 5.5, 4.5}, {-94.3345, 435.6, 94.}};
        B = new MatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(13.45), new CNumber(5.5), new CNumber(4.5)},
                {new CNumber(-94.3345), new CNumber(435.6), new CNumber(94.)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realSparseMatrixTestCase() {
        double[] bEntries;
        CooMatrixOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), CNumber.ZERO, new CNumber(1.234), CNumber.ZERO},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(345, 1);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {CNumber.ZERO, new CNumber(1.234), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(234, 1233);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexMatrixTestCase() {
        CNumber[][] bEntries;
        CMatrixOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234)}, {new CNumber(-1867.4, 77.51)}};
        B = new CMatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(234.5, -87.234)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-1867.4, 77.51)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber(43.566920234, 234.5)}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51), new CNumber(9, -987.43)}};
        B = new CMatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51), new CNumber(9, -987.43)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51)}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51)}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexSparseMatrixTestCase() {
        CNumber[] bEntries;
        CooCMatrixOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), CNumber.ZERO, new CNumber(-8324.324, 234.25), CNumber.ZERO},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(345, 1);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrixOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {CNumber.ZERO, new CNumber(-8324.324, 234.25), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(234, 1233);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrixOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrixOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realVectorTestCase() {
        double[] bEntries;
        VectorOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{12.34, -89345.5};
        B = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(12.34)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-89345.5)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{12.34, -89345.5, 3.4};
        B = new VectorOld(bEntries);

        VectorOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{12.34, -89345.5, 234.56};
        B = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(12.34), new CNumber(-89345.5), new CNumber(234.56)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{12.34, -89345.5};
        B = new VectorOld(bEntries);

        VectorOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{12.34, -89345.5};
        B = new VectorOld(bEntries);

        VectorOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexVectorTestCase() {
        CNumber[] bEntries;
        CVectorOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123)};
        B = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(943.5, -85.4)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-4.3, 50.123)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123), new CNumber(985.355, 634634.202)};
        B = new CVectorOld(bEntries);

        CVectorOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123), new CNumber(985.355, 634634.202)};
        B = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123), new CNumber(985.355, 634634.202)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123)};
        B = new CVectorOld(bEntries);

        CVectorOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123)};
        B = new CVectorOld(bEntries);

        CVectorOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realSparseVectorTestCase() {
        double[] bEntries;
        CooVectorOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVectorOld(2, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(1.234)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVectorOld(24, bEntries, rowIndices);

        CooVectorOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVectorOld(3, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(1.234), CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVectorOld(3546, bEntries, rowIndices);

        CooVectorOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVectorOld(3, bEntries, rowIndices);

        CooVectorOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexSparseVectorTestCase() {
        CNumber[] bEntries;
        CooCVectorOld B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVectorOld(2, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(23.45, -732.4)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVectorOld(24, bEntries, rowIndices);

        CooCVectorOld finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVectorOld(3, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(23.45, -732.4), CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVectorOld(3546, bEntries, rowIndices);

        CooCVectorOld finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVectorOld(3, bEntries, rowIndices);

        CooCVectorOld finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }
}
