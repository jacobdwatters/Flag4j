package org.flag4j.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixStackTests {
    Shape sparseShape;
    int[] rowIndices, colIndices;
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realMatrixTestCase() {
        double[][] bEntries;
        Matrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(13.45), new CNumber(5.5)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-94.3345), new CNumber(435.6)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5, 23.45}, {-94.3345, 435.6, -8234.2}, {3.67, -798.41, 45.6}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5, 4.5}, {-94.3345, 435.6, 94.}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(13.45), new CNumber(5.5), new CNumber(4.5)},
                {new CNumber(-94.3345), new CNumber(435.6), new CNumber(94.)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new Matrix(bEntries);

        Matrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new Matrix(bEntries);

        Matrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realSparseMatrixTestCase() {
        double[] bEntries;
        CooMatrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), CNumber.ZERO, new CNumber(1.234), CNumber.ZERO},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(345, 1);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {CNumber.ZERO, new CNumber(1.234), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(234, 1233);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexMatrixTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234)}, {new CNumber(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(234.5, -87.234)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-1867.4, 77.51)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(43.566920234, 234.5)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51), new CNumber(9, -987.43)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51), new CNumber(9, -987.43)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);

        CMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);

        CMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexSparseMatrixTestCase() {
        CNumber[] bEntries;
        CooCMatrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), CNumber.ZERO, new CNumber(-8324.324, 234.25), CNumber.ZERO},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(345, 1);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {CNumber.ZERO, new CNumber(-8324.324, 234.25), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(234, 1233);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realVectorTestCase() {
        double[] bEntries;
        Vector B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{12.34, -89345.5};
        B = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(12.34)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-89345.5)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{12.34, -89345.5, 3.4};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{12.34, -89345.5, 234.56};
        B = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(12.34), new CNumber(-89345.5), new CNumber(234.56)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{12.34, -89345.5};
        B = new Vector(bEntries);

        Vector finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{12.34, -89345.5};
        B = new Vector(bEntries);

        Vector finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexVectorTestCase() {
        CNumber[] bEntries;
        CVector B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123)};
        B = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(943.5, -85.4)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), new CNumber(-4.3, 50.123)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123), new CNumber(985.355, 634634.202)};
        B = new CVector(bEntries);

        CVector finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123), new CNumber(985.355, 634634.202)};
        B = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123), new CNumber(985.355, 634634.202)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123)};
        B = new CVector(bEntries);

        CVector finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(943.5, -85.4), new CNumber(-4.3, 50.123)};
        B = new CVector(bEntries);

        CVector finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realSparseVectorTestCase() {
        double[] bEntries;
        CooVector B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVector(2, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(1.234)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVector(24, bEntries, rowIndices);

        CooVector finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVector(3, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(1.234), CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVector(3546, bEntries, rowIndices);

        CooVector finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        B = new CooVector(3, bEntries, rowIndices);

        CooVector finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexSparseVectorTestCase() {
        CNumber[] bEntries;
        CooCVector B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVector(2, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(23.45, -732.4)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3), CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVector(24, bEntries, rowIndices);

        CooCVector finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVector(3, bEntries, rowIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)},
                {new CNumber(23.45, -732.4), CNumber.ZERO, CNumber.ZERO}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVector(3546, bEntries, rowIndices);

        CooCVector finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), CNumber.ZERO, new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(23.45, -732.4)};
        rowIndices = new int[]{0};
        B = new CooCVector(3, bEntries, rowIndices);

        CooCVector finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }
}
