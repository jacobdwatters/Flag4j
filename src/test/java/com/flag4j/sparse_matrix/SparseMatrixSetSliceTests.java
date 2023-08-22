package com.flag4j.sparse_matrix;

import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseMatrixSetSliceTests {

    @Test
    void setSliceTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        SparseMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        SparseMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        SparseMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.26601, 0.19614, 0.44447};
        aRowIndices = new int[]{0, 0, 3};
        aColIndices = new int[]{0, 2, 0};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.04135, 0.54584};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{2, 0};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.26601, 0.19614, 0.04135, 0.54584};
        expRowIndices = new int[]{0, 0, 2, 3};
        expColIndices = new int[]{0, 2, 2, 0};
        exp = new SparseMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.8099, 0.98458, 0.86365, 0.22484, 0.12245};
        aRowIndices = new int[]{4, 7, 13, 21, 22};
        aColIndices = new int[]{8, 2, 10, 8, 6};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 9);
        bEntries = new double[]{0.32305, 0.32692, 0.79176, 0.19365, 0.52579, 0.74421, 0.28361, 0.49306, 0.97286, 0.93381, 0.32774, 0.92891, 0.72314, 0.30798, 0.60839, 0.60584};
        bRowIndices = new int[]{0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4};
        bColIndices = new int[]{3, 7, 0, 1, 5, 6, 3, 6, 7, 8, 1, 3, 6, 7, 0, 4};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.8099, 0.98458, 0.86365, 0.32305, 0.32692, 0.79176, 0.19365, 0.52579, 0.74421, 0.28361, 0.49306, 0.97286, 0.93381, 0.32774, 0.92891, 0.72314, 0.30798, 0.60839, 0.60584};
        expRowIndices = new int[]{4, 7, 13, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22};
        expColIndices = new int[]{8, 2, 10, 4, 8, 1, 2, 6, 7, 4, 7, 8, 9, 2, 4, 7, 8, 1, 5};
        exp = new SparseMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.66147, 0.39264, 0.46927, 0.54339, 0.8033, 0.4519, 0.39398, 0.46898, 0.78049};
        aRowIndices = new int[]{41, 45, 141, 341, 376, 480, 522, 644, 828};
        aColIndices = new int[]{2, 1, 4, 2, 2, 2, 4, 1, 3};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new double[]{0.11246, 0.6844};
        bRowIndices = new int[]{1, 2};
        bColIndices = new int[]{1, 0};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.11246, 0.6844, 0.66147, 0.39264, 0.46927, 0.54339, 0.8033, 0.4519, 0.39398, 0.46898, 0.78049};
        expRowIndices = new int[]{1, 2, 41, 45, 141, 341, 376, 480, 522, 644, 828};
        expColIndices = new int[]{1, 0, 2, 1, 4, 2, 2, 2, 4, 1, 3};
        exp = new SparseMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.00945, 0.0801, 0.88807, 0.02644};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 0, 3, 4};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.05356};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseMatrix final0a = a;
        SparseMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.59692, 0.47247, 0.26357, 0.74655};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{0, 4, 2, 4};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 1);
        bEntries = new double[]{0.20907};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseMatrix final1a = a;
        SparseMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.15919, 0.18619, 0.9727, 0.67348};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 3, 2, 4};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 2);
        bEntries = new double[]{0.81163};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseMatrix final2a = a;
        SparseMatrix final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.11608, 0.45899, 0.18176, 0.52675};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 2, 0};
        a = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.88384};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseMatrix final3a = a;
        SparseMatrix final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }

}
