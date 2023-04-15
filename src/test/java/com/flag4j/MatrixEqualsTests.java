package com.flag4j;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixEqualsTests {
    double[][] aEntries;
    CNumber[][] bEntries;

    Shape cShape;
    double[] cEntries;
    int[][] cIndices;

    Shape dShape;
    CNumber[] dEntries;
    int[][] dIndices;

    boolean exp;
    Matrix A;
    CMatrix B;
    SparseMatrix C;
    SparseCMatrix D;

    @Test
    void matrixCMatrixEqualsTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}};
        bEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(4)},
                {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(7)},
                {new CNumber(7), new CNumber(8), new CNumber(9), new CNumber(10)}};
        A = new Matrix(aEntries);
        B = new CMatrix(bEntries);
        exp = true;

        assertEquals(exp, A.equals(B));

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}};
        bEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(4.12)},
                        {new CNumber(4), new CNumber(5), new CNumber(6), new CNumber(7)},
                        {new CNumber(7), new CNumber(8), new CNumber(9), new CNumber(10)}};
        A = new Matrix(aEntries);
        B = new CMatrix(bEntries);
        exp = false;

        assertEquals(exp, A.equals(B));

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{1, 2}, {4, 5}, {7, 8}};
        bEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2)},
                        {new CNumber(4), new CNumber(5)},
                        {new CNumber(7), new CNumber(8)}};
        A = new Matrix(aEntries);
        B = new CMatrix(bEntries);
        exp = true;

        assertEquals(exp, A.equals(B));

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{1, 2}, {4, 5}, {7, 8}};
        bEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2, 1)},
                        {new CNumber(4), new CNumber(5)},
                        {new CNumber(7), new CNumber(8)}};
        A = new Matrix(aEntries);
        B = new CMatrix(bEntries);
        exp = false;

        assertEquals(exp, A.equals(B));
    }


    @Test
    void matrixSparseMatrixEqualsTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        cShape = new Shape(aEntries.length, aEntries[0].length);
        cEntries = new double[]{1, 6, 8, 9, 10};
        cIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        C = new SparseMatrix(cShape, cEntries, cIndices[0], cIndices[1]);

        exp = true;

        assertEquals(exp, A.equals(C));

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, 0, 0, 1}, {0, 0, 6, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        cShape = new Shape(aEntries.length, aEntries[0].length);
        cEntries = new double[]{1, 6, 8, 9, 10};
        cIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        C = new SparseMatrix(cShape, cEntries, cIndices[0], cIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(C));

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        cShape = new Shape(aEntries.length, aEntries[0].length);
        cEntries = new double[]{1, -8, 8, 9, 10};
        cIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        C = new SparseMatrix(cShape, cEntries, cIndices[0], cIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(C));

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        cShape = new Shape(aEntries.length, aEntries[0].length);
        cEntries = new double[]{1, 4, 6, 8, 9, 10};
        cIndices = new int[][]{{0, 0, 1, 2, 2, 2}, {0, 3, 2, 1, 2, 3}};
        C = new SparseMatrix(cShape, cEntries, cIndices[0], cIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(C));

        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}, {0, 0, 0, 0}};
        A = new Matrix(aEntries);
        cShape = new Shape(3, 4);
        cEntries = new double[]{1, 6, 8, 9, 10};
        cIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        C = new SparseMatrix(cShape, cEntries, cIndices[0], cIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(C));
    }


    @Test
    void matrixSparseCMatrixEqualsTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        dShape = new Shape(aEntries.length, aEntries[0].length);
        dEntries = new CNumber[]{new CNumber(1), new CNumber(6), new CNumber(8),
                new CNumber(9), new CNumber(10)};
        dIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        D = new SparseCMatrix(dShape, dEntries, dIndices[0], dIndices[1]);

        exp = true;

        assertEquals(exp, A.equals(D));

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        dShape = new Shape(aEntries.length, aEntries[0].length);
        dEntries = new CNumber[]{new CNumber(1), new CNumber(6, 1), new CNumber(8),
                new CNumber(9), new CNumber(10)};
        dIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        D = new SparseCMatrix(dShape, dEntries, dIndices[0], dIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(D));

        // -------------- Sub-case 3 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, -9.34, 10}};
        A = new Matrix(aEntries);
        dShape = new Shape(aEntries.length, aEntries[0].length);
        dEntries = new CNumber[]{new CNumber(1), new CNumber(6), new CNumber(8),
                new CNumber(9), new CNumber(10)};
        dIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        D = new SparseCMatrix(dShape, dEntries, dIndices[0], dIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(D));

        // -------------- Sub-case 4 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}, {0, 0, 0, 0}};
        A = new Matrix(aEntries);
        dShape = new Shape(aEntries.length, aEntries[0].length);
        dEntries = new CNumber[]{new CNumber(1), new CNumber(6), new CNumber(8),
                new CNumber(9), new CNumber(10)};
        dIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        D = new SparseCMatrix(dShape, dEntries, dIndices[0], dIndices[1]);

        exp = true;

        assertEquals(exp, A.equals(D));

        // -------------- Sub-case 5 --------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 0, 0, 0}, {0, 8, 9, 10}};
        A = new Matrix(aEntries);
        dShape = new Shape(aEntries.length, aEntries[0].length);
        dEntries = new CNumber[]{new CNumber(1), new CNumber(6), new CNumber(8),
                new CNumber(9), new CNumber(10)};
        dIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        D = new SparseCMatrix(dShape, dEntries, dIndices[0], dIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(D));

        // ----------------- Sub-case 6 -----------------
        aEntries = new double[][]{{1, 0, 0, 0}, {0, 0, 6, 0}, {0, 8, 9, 10}, {0, 0, 0, 0}};
        A = new Matrix(aEntries);
        dShape = new Shape(3, 4);
        dEntries = new CNumber[]{new CNumber(1), new CNumber(6), new CNumber(8),
                new CNumber(9), new CNumber(10)};
        dIndices = new int[][]{{0, 1, 2, 2, 2}, {0, 2, 1, 2, 3}};
        D = new SparseCMatrix(dShape, dEntries, dIndices[0], dIndices[1]);

        exp = false;

        assertEquals(exp, A.equals(D));
    }
}
