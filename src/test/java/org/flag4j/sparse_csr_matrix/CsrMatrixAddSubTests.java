package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixAddSubTests {

    static CsrMatrix A;
    static CsrMatrix B;
    static Matrix denseA;
    static Matrix denseB;
    static CsrMatrix expAdd;
    static CsrMatrix expAsubB;
    static CsrMatrix expBsubA;
    static Matrix expAddDense;
    static Matrix expAsubBDense;
    static Matrix expBsubADense;
    static double[][] aEntries;
    static double[][] bEntries;

    static CNumber[][] bCmpEntries;
    static CsrCMatrix BCmp;
    static CsrCMatrix expAddCmp;
    static CsrCMatrix expAsubBCmp;
    static CMatrix expAddDenseCmp;
    static CMatrix expAsubBDenseCmp;
    static CMatrix denseBCmp;
    static double b;
    static CNumber bCmp;

    private static void makeRealMatrices() {
        denseA = new Matrix(aEntries);
        denseB = new Matrix(bEntries);
        A = denseA.toCsr();
        B = denseB.toCsr();
        expAddDense = denseA.add(denseB);
        expAsubBDense = denseA.sub(denseB);
        expBsubADense = denseB.sub(denseA);
        expAdd = expAddDense.toCsr();
        expAsubB = expAsubBDense.toCsr();
        expBsubA = expBsubADense.toCsr();
    }

    private static void makeRealConstMatrices() {
        denseA = new Matrix(aEntries);
        A = denseA.toCsr();
        expAddDense = denseA.add(b);
        expAsubBDense = denseA.sub(b);
    }

    private static void makeCmpMatrices() {
        denseA = new Matrix(aEntries);
        denseBCmp = new CMatrix(bCmpEntries);
        A = denseA.toCsr();
        BCmp = denseBCmp.toCsr();
        expAddDenseCmp = denseA.add(denseBCmp);
        expAsubBDenseCmp = denseA.sub(denseBCmp);
        expAddCmp = expAddDenseCmp.toCsr();
        expAsubBCmp = expAsubBDenseCmp.toCsr();
    }

    private static void makeCmpConstMatrices() {
        denseA = new Matrix(aEntries);
        A = denseA.toCsr();
        expAddDenseCmp = denseA.add(bCmp);
        expAsubBDenseCmp = denseA.sub(bCmp);
    }


    @Test
    void addSubSpTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {0, 0, 1.34, 0, 1.3, 0},
                {0, 0, 0, 56.1, 0, 0},
                {36.1, 13.2, 0, 0, 0, 8},
                {0, 0, 0, 0, 0, 0},
        };
        makeRealMatrices();
        assertEquals(expAdd, A.add(B));
        assertEquals(expAsubB, A.sub(B));
        assertEquals(expBsubA, B.sub(A));

        assertEquals(expAdd, A.add(B.toCoo()));
        assertEquals(expAsubB, A.sub(B.toCoo()));
        assertEquals(expBsubA, B.sub(A.toCoo()));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
        assertThrows(LinearAlgebraException.class, ()->B.add(A));
        assertThrows(LinearAlgebraException.class, ()->B.sub(A));

        assertThrows(LinearAlgebraException.class, ()->A.add(B.toCoo()));
        assertThrows(LinearAlgebraException.class, ()->A.sub(B.toCoo()));
        assertThrows(LinearAlgebraException.class, ()->B.add(A.toCoo()));
        assertThrows(LinearAlgebraException.class, ()->B.sub(A.toCoo()));
    }


    @Test
    void addSubDeTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bEntries = new double[][]{
                {1, 2, 4, 25.23, 1.5, -0.204},
                {-99.34, 15.23, 1.34, 0, 1.3, 0},
                {50.2, 3491.3, 2.42, 56.1, Double.POSITIVE_INFINITY, 0},
                {36.1, 13.2, 2.5621, 83458934.245, 0, 8},
                {1.345, 983.3, 0, -9.234, 4.52, 0},
        };
        makeRealMatrices();
        assertEquals(expAddDense, A.add(denseB));
        assertEquals(expAsubBDense, A.sub(denseB));
        assertEquals(expBsubADense, B.sub(denseA));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(B));
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
        assertThrows(LinearAlgebraException.class, ()->B.add(A));
        assertThrows(LinearAlgebraException.class, ()->B.sub(A));
    }


    @Test
    void addSubSpCmpTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmpEntries = new CNumber[aEntries.length][aEntries[0].length];
        ArrayUtils.fillZeros(bCmpEntries);
        bCmpEntries[0][0] = new CNumber(23, 1.34);
        bCmpEntries[1][0] = new CNumber(0.133, -41.4);
        bCmpEntries[1][3] = new CNumber(-4.1, -34.1);
        bCmpEntries[3][1] = new CNumber(922.1);
        bCmpEntries[3][5] = new CNumber(34.5, 135);
        bCmpEntries[4][4] = new CNumber(23.501, 100.23);
        makeCmpMatrices();
        assertEquals(expAddCmp, A.add(BCmp));
        assertEquals(expAsubBCmp, A.sub(BCmp));
        assertEquals(expAddCmp, A.add(BCmp.toCoo()));
        assertEquals(expAsubBCmp, A.sub(BCmp.toCoo()));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(BCmp));
        assertThrows(LinearAlgebraException.class, ()->A.sub(BCmp));
        assertThrows(LinearAlgebraException.class, ()->A.add(BCmp.toCoo()));
        assertThrows(LinearAlgebraException.class, ()->A.sub(BCmp.toCoo()));
    }


    @Test
    void addSubDeCmpTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmpEntries = new CNumber[aEntries.length][aEntries[0].length];
        ArrayUtils.fillZeros(bCmpEntries);
        bCmpEntries[0][0] = new CNumber(23, 1.34);
        bCmpEntries[1][0] = new CNumber(0.133, -41.4);
        bCmpEntries[1][3] = new CNumber(-4.1, -34.1);
        bCmpEntries[3][1] = new CNumber(922.1);
        bCmpEntries[3][5] = new CNumber(34.5, 135);
        bCmpEntries[4][4] = new CNumber(23.501, 100.23);
        makeCmpMatrices();
        assertEquals(expAddDenseCmp, A.add(denseBCmp));
        assertEquals(expAsubBDenseCmp, A.sub(denseBCmp));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrMatrix(new Shape(2, 3), new double[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(denseBCmp));
        assertThrows(LinearAlgebraException.class, ()->A.sub(denseBCmp));
    }


    @Test
    void addSubRealConstTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        b = 1.2;
        makeRealConstMatrices();

        assertEquals(expAddDense, A.add(b));
        assertEquals(expAsubBDense, A.sub(b));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        b = -2341.200239867181;
        makeRealConstMatrices();

        assertEquals(expAddDense, A.add(b));
        assertEquals(expAsubBDense, A.sub(b));
    }


    @Test
    void addSubCmpConstTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmp = new CNumber(-0.234, 155.2);
        makeCmpConstMatrices();

        assertEquals(expAddDenseCmp, A.add(bCmp));
        assertEquals(expAsubBDenseCmp, A.sub(bCmp));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {1.2324, 0, 0, 13.4, 0, 0},
                {0, 0, 0, -23.5, 0, 0},
                {0, 14.1, 0, 0, 0, 0},
                {0, 0, 0, 9.143, 1.4, -2.1}};
        bCmp = new CNumber(-92.14, -7884.7761);
        makeCmpConstMatrices();

        assertEquals(expAddDenseCmp, A.add(bCmp));
        assertEquals(expAsubBDenseCmp, A.sub(bCmp));
    }
}
