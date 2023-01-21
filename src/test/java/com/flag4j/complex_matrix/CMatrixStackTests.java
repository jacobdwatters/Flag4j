package com.flag4j.complex_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CMatrixStackTests {
    Shape sparseShape;
    int[] rowIndices, colIndices;
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realMatrixTest() {
        double[][] bEntries;
        Matrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(13.45), new CNumber(5.5)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3), new CNumber(-94.3345), new CNumber(435.6)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5, 23.45}, {-94.3345, 435.6, -8234.2}, {3.67, -798.41, 45.6}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5, 4.5}, {-94.3345, 435.6, 94.}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)},
                {new CNumber(13.45), new CNumber(5.5), new CNumber(4.5)},
                {new CNumber(-94.3345), new CNumber(435.6), new CNumber(94.)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new Matrix(bEntries);

        Matrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{13.45, 5.5}, {-94.3345, 435.6}};
        B = new Matrix(bEntries);

        Matrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void realSparseMatrixTest() {
        double[] bEntries;
        SparseMatrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(), new CNumber(1.234), new CNumber()},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3), new CNumber(), new CNumber(), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(345, 1);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)},
                {new CNumber(), new CNumber(1.234), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(234, 1233);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.234};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexMatrixTest() {
        CNumber[][] bEntries;
        CMatrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234)}, {new CNumber(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(234.5, -87.234)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3), new CNumber(-1867.4, 77.51)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(43.566920234, 234.5)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51), new CNumber(9, -987.43)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)},
                {new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51), new CNumber(9, -987.43)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);

        CMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber(234.5, -87.234), new CNumber(-1867.4, 77.51)}};
        B = new CMatrix(bEntries);

        CMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }


    @Test
    void complexSparseMatrixTest() {
        CNumber[] bEntries;
        SparseCMatrix B;

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3), new CNumber(), new CNumber(-8324.324, 234.25), new CNumber()},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3), new CNumber(), new CNumber(), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 0));

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(345, 1);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseCMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB, 0));

        // ----------------------- Sub-case 3 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)},
                {new CNumber(), new CNumber(-8324.324, 234.25), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.stack(B, 1));

        // ----------------------- Sub-case 4 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(234, 1233);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseCMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 1));

        // ----------------------- Sub-case 5 -----------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(-8324.324, 234.25)};
        rowIndices = new int[]{0};
        colIndices = new int[]{1};
        sparseShape = new Shape(2, 3);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseCMatrix finalB2 = B;
        assertThrows(IllegalArgumentException.class, ()->A.stack(finalB1, 2));
    }
}
