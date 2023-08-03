package com.flag4j.sparse_matrix;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@SuppressWarnings("AssertBetweenInconvertibleTypes")
class SparseMatrixEqualsTest {

    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;
    static SparseMatrix A;


    @BeforeAll
    static void setup() {
        aShape = new Shape(401, 13_440);
        aEntries = new double[]{1.34, 100.14, -9.245, 0.00234, 52.5};
        aIndices = new int[][]{
                {9, 13, 141, 141, 398},
                {1_002, 5, 41, 12_234, 9_013}
        };

        A = new SparseMatrix(aShape, aEntries, aIndices[0], aIndices[1]);
    }


    @Test
    void denseEqualsTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new double[aShape.get(0)][aShape.get(1)];
        fillDense(bEntries);
        B = new Matrix(bEntries);

        assertEquals(A, B);

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[aShape.get(0)-1][aShape.get(1)+13];
        fillDense(bEntries);
        B = new Matrix(bEntries);

        assertNotEquals(A, B);

        // --------------------- Sub-case 3 ---------------------
        bEntries = new double[aShape.get(0)][aShape.get(1)];
        fillDense(bEntries);
        bEntries[134][7624] = -1;
        B = new Matrix(bEntries);

        assertNotEquals(A, B);

        // --------------------- Sub-case 4 ---------------------
        bEntries = new double[aShape.get(0)][aShape.get(1)];
        fillDense(bEntries);
        bEntries[141][41] = 0;
        B = new Matrix(bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void denseComplexEqualsTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new CNumber[aShape.get(0)][aShape.get(1)];
        ArrayUtils.fill(bEntries, CNumber.ZERO);
        fillDense(bEntries);
        B = new CMatrix(bEntries);

        assertEquals(A, B);

        // --------------------- Sub-case 2 ---------------------
        bEntries = new CNumber[aShape.get(0)-1][aShape.get(1)+13];
        ArrayUtils.fill(bEntries, CNumber.ZERO);
        fillDense(bEntries);
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);

        // --------------------- Sub-case 3 ---------------------
        bEntries = new CNumber[aShape.get(0)][aShape.get(1)];
        ArrayUtils.fill(bEntries, CNumber.ZERO);
        fillDense(bEntries);
        bEntries[134][7624] = new CNumber(0, -0.3);
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);

        // --------------------- Sub-case 4 ---------------------
        bEntries = new CNumber[aShape.get(0)][aShape.get(1)];
        ArrayUtils.fill(bEntries, CNumber.ZERO);
        fillDense(bEntries);
        bEntries[141][41] = new CNumber();
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);
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
