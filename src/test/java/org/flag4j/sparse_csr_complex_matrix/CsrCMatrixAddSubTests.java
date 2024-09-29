package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.operations.dense_sparse.csr.complex.ComplexCsrDenseOperations;
import org.flag4j.operations.dense_sparse.csr.real_complex.RealComplexCsrDenseOperations;
import org.flag4j.operations.sparse.csr.real_complex.RealComplexCsrOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class CsrCMatrixAddSubTests {
    static CsrCMatrix A;
    static CsrMatrix B;
    static CMatrix denseA;
    static Matrix denseB;
    static CsrCMatrix expAdd;
    static CsrCMatrix expAsubB;
    static CsrCMatrix expBsubA;
    static CMatrix expAddDense;
    static CMatrix expAsubBDense;
    static CMatrix expBsubADense;
    static Complex128[][] aEntries;
    static double[][] bEntries;

    static double b;
    static Complex128 bCmp;
    static Complex128[][] bEntriesCmp;
    static CsrCMatrix BCmp;
    static CMatrix denseBCmp;

    private static void makeMatrices() {
        denseA = new CMatrix(aEntries);
        A = denseA.toCsr();

        if(bEntries != null) {
            denseB = new Matrix(bEntries);
            B = denseB.toCsr();
            expAddDense = denseA.add(denseB);
            expAsubBDense = denseA.sub(denseB);
            expBsubADense = denseB.sub(denseA);
            expAdd = expAddDense.toCsr();
            expAsubB = expAsubBDense.toCsr();
            expBsubA = expBsubADense.toCsr();
        } else if(bEntriesCmp != null) {
            denseBCmp = new CMatrix(bEntriesCmp);
            BCmp = denseBCmp.toCsr();
            expAddDense = denseA.add(denseBCmp);
            expAsubBDense = denseA.sub(denseBCmp);
            expBsubADense = denseBCmp.sub(denseA);
            expAdd = expAddDense.toCsr();
            expAsubB = expAsubBDense.toCsr();
            expBsubA = expBsubADense.toCsr();
        }
    }

    private static void makeRealConstMatrices() {
        denseA = new CMatrix(aEntries);
        A = denseA.toCsr();
        expAddDense = denseA.add(b);
        expAsubBDense = denseA.sub(b);
    }

    private static void makeCmpConstMatrices() {
        denseA = new CMatrix(aEntries);
        A = denseA.toCsr();
        expAddDense = denseA.add(bCmp);
        expAsubBDense = denseA.sub(bCmp);
    }


    private static void resetAll() {
        A = null;
        B = null;
        denseA = null;
        denseB = null;
        expAdd = null;
        expAsubB = null;
        expBsubA = null;
        expAddDense = null;
        expAsubBDense = null;
        expBsubADense = null;
        aEntries = null;
        bEntries = null;

        bEntriesCmp = null;
        BCmp = null;
        denseBCmp = null;
    }


    @Test
    void addSubSpTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        bEntries = new double[][]{
                {0, 0, 0, 0, 0, 0},
                {0, 0, 1.34, 0, 1.3, 0},
                {0, 0, 0, 56.1, 0, 0},
                {36.1, 13.2, 0, 0, 0, 8},
                {0, 0, 0, 0, 0, 0},
        };
        makeMatrices();
        assertEquals(expAdd, RealComplexCsrOperations.add(A, B));
        assertEquals(expAsubB, RealComplexCsrOperations.sub(A, B));
        assertEquals(expBsubA, RealComplexCsrOperations.sub(B, A));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrOperations.add(A, B));
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrOperations.sub(A, B));
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrOperations.sub(B, A));

        resetAll();
    }


    @Test
    void addSubDeTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        bEntries = new double[][]{
                {1, 2, 4, 25.23, 1.5, -0.204},
                {-99.34, 15.23, 1.34, 0, 1.3, 0},
                {50.2, 3491.3, 2.42, 56.1, Double.POSITIVE_INFINITY, 0},
                {36.1, 13.2, 2.5621, 83458934.245, 0, 8},
                {1.345, 983.3, 0, -9.234, 4.52, 0},
        };
        makeMatrices();
        assertEquals(expAddDense, RealComplexCsrDenseOperations.add(A, denseB));
        assertEquals(expAsubBDense, RealComplexCsrDenseOperations.sub(A, denseB));
        assertEquals(expBsubADense, RealComplexCsrDenseOperations.sub(B, denseA));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrDenseOperations.add(A, denseB));
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrDenseOperations.sub(A, denseB));
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrDenseOperations.add(B, denseA));
        assertThrows(LinearAlgebraException.class, ()->RealComplexCsrDenseOperations.sub(B, denseA));

        resetAll();
    }


    @Test
    void addSubSpCmpTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        bEntriesCmp = new Complex128[aEntries.length][aEntries[0].length];
        ArrayUtils.fill(bEntriesCmp, Complex128.ZERO);
        bEntriesCmp[0][0] = new Complex128(23, 1.34);
        bEntriesCmp[1][0] = new Complex128(0.133, -41.4);
        bEntriesCmp[1][3] = new Complex128(-4.1, -34.1);
        bEntriesCmp[3][1] = new Complex128(922.1);
        bEntriesCmp[3][5] = new Complex128(34.5, 135);
        bEntriesCmp[4][4] = new Complex128(23.501, 100.23);
        makeMatrices();

        assertEquals(expAdd, A.add(BCmp));
        assertEquals(expAsubB, A.sub(BCmp));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->A.add(BCmp));
        assertThrows(LinearAlgebraException.class, ()->A.sub(BCmp));

        resetAll();
    }


    @Test
    void addSubDeCmpTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        bEntriesCmp = new Complex128[aEntries.length][aEntries[0].length];
        ArrayUtils.fill(bEntriesCmp, Complex128.ZERO);
        bEntriesCmp[0][0] = new Complex128(23, 1.34);
        bEntriesCmp[1][0] = new Complex128(0.133, -41.4);
        bEntriesCmp[1][3] = new Complex128(-4.1, -34.1);
        bEntriesCmp[3][1] = new Complex128(922.1);
        bEntriesCmp[3][5] = new Complex128(34.5, 135);
        bEntriesCmp[4][4] = new Complex128(23.501, 100.23);
        makeMatrices();
        assertEquals(expAddDense, ComplexCsrDenseOperations.add(A, denseBCmp));
        assertEquals(expAsubBDense, ComplexCsrDenseOperations.sub(A, denseBCmp));

        // ---------------------- Sub-case 2 ----------------------
        A = new CsrCMatrix(new Shape(2, 3), new Complex128[0], new int[3], new int[0]);
        B = new CsrMatrix(new Shape(5, 1), new double[0], new int[6], new int[0]);
        assertThrows(LinearAlgebraException.class, ()->ComplexCsrDenseOperations.add(A, denseBCmp));
        assertThrows(LinearAlgebraException.class, ()->ComplexCsrDenseOperations.sub(A, denseBCmp));

        resetAll();
    }


    @Test
    void addSubRealConstTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        b = 1.2;
        makeRealConstMatrices();

        assertEquals(expAddDense, A.add(b));
        assertEquals(expAsubBDense, A.sub(b));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        b = -2341.200239867181;
        makeRealConstMatrices();

        assertEquals(expAddDense, A.add(b));
        assertEquals(expAsubBDense, A.sub(b));
    }


    @Test
    void addSubCmpConstTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        bCmp = new Complex128(-0.234, 155.2);
        makeCmpConstMatrices();

        assertEquals(expAddDense, A.add(bCmp));
        assertEquals(expAsubBDense, A.sub(bCmp));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.6947844604184964+0.12617739377846993i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.5801548358148666+0.26523862055012315i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.030397833484514858+0.0339598081443111i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.8303067369813061+0.6672665799918752i"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.08364434412041977+0.4996667151892932i"), new Complex128("0.4271149460271625+0.6885614097388274i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        bCmp = new Complex128(-92.14, -7884.7761);
        makeCmpConstMatrices();

        assertEquals(expAddDense, A.add(bCmp));
        assertEquals(expAsubBDense, A.sub(bCmp));
    }
}
