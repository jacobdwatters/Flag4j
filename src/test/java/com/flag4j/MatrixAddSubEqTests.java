package com.flag4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class MatrixAddSubEqTests {

    int[] rowIndices, colIndices;
    Shape sparseShape;

    double[][] aEntries;
    Matrix A, exp;

    @Test
    void addEqRealTest() {
        double[][] bEntries;
        Matrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634, 14.466}, {Double.POSITIVE_INFINITY, 345.6, 8.346}};
        B = new Matrix(bEntries);
        exp = A.add(B);

        A.addEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634}, {Double.POSITIVE_INFINITY, 345.6}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.addEq(finalB));
    }


    @Test
    void addEqSparseTest() {
        double[] bEntries;
        SparseMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = new Shape(A.shape);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);
        exp = A.add(B);

        A.addEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = new Shape(5, 3);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.addEq(finalB));
    }


    @Test
    void addEqDoubleTest() {
        double b;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        b = 316.455;
        exp = A.add(b);

        A.addEq(b);

        assertEquals(exp, A);
    }


    @Test
    void subEqRealTest() {
        double[][] bEntries;
        Matrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634, 14.466}, {Double.POSITIVE_INFINITY, 345.6, 8.346}};
        B = new Matrix(bEntries);
        exp = A.sub(B);

        A.subEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634}, {Double.POSITIVE_INFINITY, 345.6}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqSparseTest() {
        double[] bEntries;
        SparseMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = new Shape(A.shape);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);
        exp = A.sub(B);

        A.subEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = new Shape(5, 3);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqDoubleTest() {
        double b;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new Matrix(aEntries);
        b = 316.455;
        exp = A.sub(b);

        A.subEq(b);

        assertEquals(exp, A);
    }
}