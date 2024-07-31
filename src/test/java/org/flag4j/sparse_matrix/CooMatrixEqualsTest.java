package org.flag4j.sparse_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

@SuppressWarnings("AssertBetweenInconvertibleTypes")
class CooMatrixEqualsTest {

    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;
    static CooMatrix A;


    @BeforeAll
    static void setup() {
        aShape = new Shape(401, 13_440);
        aEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5};
        aIndices = new int[][]{
                {9, 13, 141, 141, 398},
                {1_002, 5, 41, 12_234, 9_013}
        };

        A = new CooMatrix(aShape, aEntries, aIndices[0], aIndices[1]);
    }


    @Test
    void denseEqualsTest() {
        double[][] bEntries;
        Matrix B;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new double[aShape.get(0)][aShape.get(1)];
        fillDense(bEntries);
        B = new Matrix(bEntries);

        assertTrue(A.tensorEquals(B));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[aShape.get(0)-1][aShape.get(1)+13];
        fillDense(bEntries);
        B = new Matrix(bEntries);

        assertFalse(A.tensorEquals(B));

        // --------------------- Sub-case 3 ---------------------
        bEntries = new double[aShape.get(0)][aShape.get(1)];
        fillDense(bEntries);
        bEntries[134][7624] = -1;
        B = new Matrix(bEntries);

        assertFalse(A.tensorEquals(B));

        // --------------------- Sub-case 4 ---------------------
        bEntries = new double[aShape.get(0)][aShape.get(1)];
        fillDense(bEntries);
        bEntries[141][41] = 0;
        B = new Matrix(bEntries);

        assertFalse(A.tensorEquals(B));
    }


    @Test
    void denseComplexEqualsTest() {
        CNumber[][] bEntries;
        CMatrix B;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new CNumber[aShape.get(0)][aShape.get(1)];
        ArrayUtils.fill(bEntries, CNumber.zero());
        fillDense(bEntries);
        B = new CMatrix(bEntries);

        assertTrue(A.tensorEquals(B));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new CNumber[aShape.get(0)-1][aShape.get(1)+13];
        ArrayUtils.fill(bEntries, CNumber.zero());
        fillDense(bEntries);
        B = new CMatrix(bEntries);

        assertFalse(A.tensorEquals(B));

        // --------------------- Sub-case 3 ---------------------
        bEntries = new CNumber[aShape.get(0)][aShape.get(1)];
        ArrayUtils.fill(bEntries, CNumber.zero());
        fillDense(bEntries);
        bEntries[134][7624] = new CNumber(0, -0.3);
        B = new CMatrix(bEntries);

        assertFalse(A.tensorEquals(B));

        // --------------------- Sub-case 4 ---------------------
        bEntries = new CNumber[aShape.get(0)][aShape.get(1)];
        ArrayUtils.fill(bEntries, CNumber.zero());
        fillDense(bEntries);
        bEntries[141][41] = new CNumber();
        B = new CMatrix(bEntries);

        assertFalse(A.tensorEquals(B));
    }


    @Test
    void sparseEqualsTest() {
        double[] bEntries;
        int[][] bIndices;
        CooMatrix B;

        // --------------------- Sub-case 1 ---------------------
        B = A.copy();
        assertEquals(A, B);

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5, 24.5};
        bIndices = new int[][]{
                {9, 13, 141, 141, 398, 400},
                {1_002, 5, 41, 12_234, 9_013, 27}
        };
        B = new CooMatrix(A.shape, bEntries, bIndices[0], bIndices[1]);
        assertNotEquals(A, B);

        // --------------------- Sub-case 3 ---------------------
        bEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5};
        bIndices = new int[][]{
                {9, 13, 141, 141, 398},
                {1_002, 5, 41, 12_234, 9_013}
        };
        B = new CooMatrix(new Shape(2451, 134415), bEntries, bIndices[0], bIndices[1]);
        assertNotEquals(A, B);
    }


    @Test
    void sparseComplexEqualsTest() {
        double[] bEntries;
        int[][] bIndices;
        CooCMatrix B;

        // --------------------- Sub-case 1 ---------------------
        B = A.toComplex();
        assertTrue(A.tensorEquals(B));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5, 24.5};
        bIndices = new int[][]{
                {9, 13, 141, 141, 398, 400},
                {1_002, 5, 41, 12_234, 9_013, 27}
        };
        B = new CooCMatrix(A.shape, bEntries, bIndices[0], bIndices[1]);
        assertFalse(A.tensorEquals(B));

        // --------------------- Sub-case 3 ---------------------
        bEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5};
        bIndices = new int[][]{
                {9, 13, 141, 141, 398},
                {1_002, 5, 41, 12_234, 9_013}
        };
        B = new CooCMatrix(new Shape(2451, 134415), bEntries, bIndices[0], bIndices[1]);
        assertFalse(A.tensorEquals(B));
    }


    private void fillDense(double[][] arr) {
        for(int i=0; i<aEntries.length; i++) {
            arr[A.rowIndices[i]][A.colIndices[i]] = aEntries[i];
        }
    }


    private void fillDense(CNumber[][] arr) {
        for(int i=0; i<aEntries.length; i++) {
            arr[A.rowIndices[i]][A.colIndices[i]] = new CNumber(aEntries[i]);
        }
    }
}
