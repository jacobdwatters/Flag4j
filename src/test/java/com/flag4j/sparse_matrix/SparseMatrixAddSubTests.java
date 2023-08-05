package com.flag4j.sparse_matrix;

import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SparseMatrixAddSubTests {
    Shape aShape;
    double[] aEntries;
    int[][] aIndices;
    SparseMatrix A;

    Shape bShape, expShape;
    int[][] bIndices, expIndices;

    @Test
    void realSparseAddTest() {
        double[] bEntries, expEntries;
        SparseMatrix B, exp;

        // ------------------ Sub-case 1 ------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{650.017, 138.848, -151.247, -72.509, -40.546, -656.7};
        aIndices = new int[][]{
                {0, 2, 2, 2, 2, 3},
                {0, 0, 1, 3, 4, 3}
        };
        A = new SparseMatrix(aShape, aEntries, aIndices[0], aIndices[1]);

        bShape = new Shape(5, 5);
        bEntries = new double[]{621.726, -465.916, 209.294, -802.516, -559.7, -705.709};
        bIndices = new int[][]{
                {0, 1, 2, 3, 4, 4},
                {1, 3, 4, 3, 0, 1}
        };
        B = new SparseMatrix(bShape, bEntries, bIndices[0], bIndices[1]);

        expShape = new Shape(5, 5);
        expEntries = new double[]{
                650.017, 621.726, -465.916, 138.848, -151.247, -72.509, -40.546+209.294, -656.7-802.516, -559.7, -705.709
        };
        expIndices = new int[][]{
                {0, 0, 1, 2, 2, 2, 2, 3, 4, 4},
                {0, 1, 3, 0, 1, 3, 4, 3, 0, 1}
        };
        exp = new SparseMatrix(expShape, expEntries, expIndices[0], expIndices[1]);

        assertEquals(exp, A.add(B));
    }
}
