package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class MatrixAddSubEqTests {

    int[] rowIndices, colIndices;
    Shape sparseShape;

    double[][] aEntries;
    MatrixOld A, exp;

    @Test
    void addEqRealTestCase() {
        double[][] bEntries;
        MatrixOld B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634, 14.466}, {Double.POSITIVE_INFINITY, 345.6, 8.346}};
        B = new MatrixOld(bEntries);
        exp = A.add(B);

        A.addEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634}, {Double.POSITIVE_INFINITY, 345.6}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.addEq(finalB));
    }


    @Test
    void addEqSparseTestCase() {
        double[] bEntries;
        CooMatrixOld B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = A.shape;
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        exp = A.add(B);

        A.addEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = new Shape(5, 3);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.addEq(finalB));
    }


    @Test
    void addEqDoubleTestCase() {
        double b;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        b = 316.455;
        exp = A.add(b);

        A.addEq(b);

        assertEquals(exp, A);
    }


    @Test
    void subEqRealTestCase() {
        double[][] bEntries;
        MatrixOld B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634, 14.466}, {Double.POSITIVE_INFINITY, 345.6, 8.346}};
        B = new MatrixOld(bEntries);
        exp = A.sub(B);

        A.subEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[][]{{23.46, -9346346.34634}, {Double.POSITIVE_INFINITY, 345.6}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqSparseTestCase() {
        double[] bEntries;
        CooMatrixOld B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = A.shape;
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        exp = A.sub(B);

        A.subEq(B);

        assertEquals(exp, A);

        // --------------------- Sub-case 2 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.34, -93.346};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        sparseShape = new Shape(5, 3);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqDoubleTestCase() {
        double b;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new double[][]{{1, 2.435, -843.5}, {34.56, 8.52, 0.000345}};
        A = new MatrixOld(aEntries);
        b = 316.455;
        exp = A.sub(b);

        A.subEq(b);

        assertEquals(exp, A);
    }
}
