package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixElemMultTests {

    MatrixOld A, B, result, expResult;
    CMatrixOld BC, resultC, expResultC;
    double[][] entriesA, entriesB, expEntries;
    CNumber[][] entriesBC;

    CooMatrixOld BSparse, expSparse, sparseResult;
    CooCMatrixOld BSparseComplex, expSparseComplex, sparseComplexResult;
    CNumber[] bEntriesSparseComplex, expEntriesSparseComplex;
    double[] bEntriesSparse, expEntriesSparse;
    Shape shape;

    int[] rowIndices;
    int[] colIndices;

    private double[] getExp(double[] src1, double[] src2) {
        double[] result = new double[src1.length];

        for(int i=0; i<result.length; i++) {
            result[i] = src1[i]*src2[i];
        }

        return result;
    }

    private CNumber[] getExp(double[] src1, CNumber[] src2) {
        CNumber[] result = new CNumber[src1.length];

        for(int i=0; i<result.length; i++) {
            result[i] = src2[i].mult(src1[i]);
        }

        return result;
    }


    @Test
    void elemMultTestCase() {
        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94, -0.4442}, {0.0000234, 1333.4, 44.5, 134.3}};
        A = new MatrixOld(entriesA);
        B = new MatrixOld(entriesB);
        expResult = new MatrixOld(A.shape, getExp(A.entries, B.entries));

        result = A.elemMult(B);

        assertArrayEquals(expResult.entries, result.entries);
        assertEquals(expResult.shape, result.shape);

        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94}, {0.0000234, 1333.4, 44.5}};
        A = new MatrixOld(entriesA);
        B = new MatrixOld(entriesB);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));
    }


    @Test
    void elemMultComplexTestCase() {
        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new CNumber[][]{{new CNumber(1.4, 5), new CNumber(0, -1), new CNumber(1.3), CNumber.ZERO},
                {new CNumber(4.55, -93.2), new CNumber(-2, -13), new CNumber(8.9), new CNumber(0, 13)}};
        A = new MatrixOld(entriesA);
        BC = new CMatrixOld(entriesBC);
        expResultC = new CMatrixOld(A.shape, getExp(A.entries, BC.entries));

        resultC = A.elemMult(BC);

        assertArrayEquals(expResultC.entries, resultC.entries);
        assertEquals(expResultC.shape, resultC.shape);

        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new CNumber[][]{{new CNumber(1.4, 5), new CNumber(0, -1), new CNumber(1.3)},
                {new CNumber(4.55, -93.2), new CNumber(-2, -13), new CNumber(8.9)}};
        A = new MatrixOld(entriesA);
        BC = new CMatrixOld(entriesBC);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(BC));
    }


    @Test
    void elemMultSparseTestCase() {
        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new MatrixOld(entriesA);
        bEntriesSparse = new double[]{1.45, 31.13, 7.1};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(entriesA.length, entriesA[0].length);
        BSparse = new CooMatrixOld(shape, bEntriesSparse, rowIndices, colIndices);

        expEntriesSparse = new double[]{2*1.45, -6*31.13, 0};
        expSparse = new CooMatrixOld(shape, expEntriesSparse, rowIndices.clone(), colIndices.clone());

        sparseResult = A.elemMult(BSparse);

        assertArrayEquals(expSparse.entries, sparseResult.entries);
        assertEquals(expSparse.shape, sparseResult.shape);

        // ----------------- Sub-case 2 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new MatrixOld(entriesA);
        bEntriesSparse = new double[]{1.45, 31.13, 7.1};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(3, 4);
        BSparse = new CooMatrixOld(shape, bEntriesSparse, rowIndices, colIndices);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(BSparse));
    }


    @Test
    void elemMultSparseComplexTestCase() {
        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new MatrixOld(entriesA);
        bEntriesSparseComplex = new CNumber[]{new CNumber(1000234, -8.312), new CNumber(19.334, -96.23), new CNumber(0, 1.56)};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(entriesA.length, entriesA[0].length);
        BSparseComplex = new CooCMatrixOld(shape, bEntriesSparseComplex, rowIndices, colIndices);

        expEntriesSparseComplex = new CNumber[]{bEntriesSparseComplex[0].mult(2), bEntriesSparseComplex[1].mult(-6), CNumber.ZERO};
        expSparseComplex = new CooCMatrixOld(shape, expEntriesSparseComplex, rowIndices.clone(), colIndices.clone());

        sparseComplexResult = A.elemMult(BSparseComplex);

        assertArrayEquals(expSparseComplex.entries, sparseComplexResult.entries);
        assertEquals(expSparseComplex.shape, sparseComplexResult.shape);

        // ----------------- Sub-case 2 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        A = new MatrixOld(entriesA);
        bEntriesSparseComplex = new CNumber[]{new CNumber(1000234, -8.312), new CNumber(19.334, -96.23), new CNumber(0, 1.56)};
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 2, 3};
        shape = new Shape(15, 500);
        BSparseComplex = new CooCMatrixOld(shape, bEntriesSparseComplex, rowIndices, colIndices);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(BSparseComplex));
    }
}
