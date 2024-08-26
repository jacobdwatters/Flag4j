package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixAddToEachColTests {
    double[][] aEntries, expEntries;
    double[] bEntries;
    CNumber[][] expComplexEntries;
    CNumber[] bComplexEntries;

    int[] bIndices;
    int bSize;

    MatrixOld A, exp;
    CMatrixOld expComplex;

    VectorOld b;
    CooVectorOld bSparse;
    CVectorOld bComplex;
    CooCVectorOld bSparseComplex;

    @Test
    void vectorTestCase() {
        // ---------------- Sub-case 1  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.7, -8.234, 0.1};
        b = new VectorOld(bEntries);
        expEntries = new double[][]{
                {1+1.7, 2.3+1.7},
                {-9-8.234, 13.5-8.234},
                {9.4+0.1, 0.1}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.7, -8.234};
        b = new VectorOld(bEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(b));
    }


    @Test
    void sparseVectorTestCase() {
        // ---------------- Sub-case 1  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{3.4};
        bIndices = new int[]{1};
        bSize = 3;
        bSparse = new CooVectorOld(bSize, bEntries, bIndices);
        expEntries = new double[][]{
                {1, 2.3},
                {-9+3.4, 13.5+3.4},
                {9.4, 0}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(bSparse));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{3.4};
        bIndices = new int[]{1};
        bSize = 56;
        bSparse = new CooVectorOld(bSize, bEntries, bIndices);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bSparse));
    }


    @Test
    void complexVectorTestCase() {
        // ---------------- Sub-case 1  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9.234, -.13), new CNumber(0, 5.4), new CNumber(10)};
        bComplex = new CVectorOld(bComplexEntries);
        expComplexEntries = new CNumber[][]{
                {new CNumber(1).add(bComplexEntries[0]), new CNumber(2.3).add(bComplexEntries[0])},
                {new CNumber(-9).add(bComplexEntries[1]), new CNumber(13.5).add(bComplexEntries[1])},
                {new CNumber(9.4).add(bComplexEntries[2]), new CNumber(0).add(bComplexEntries[2])}};
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, A.addToEachCol(bComplex));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9123)};
        bComplex = new CVectorOld(bComplexEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bComplex));
    }


    @Test
    void sparseComplexVectorTestCase() {
        // ---------------- Sub-case 1  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9.234, -.13)};
        bIndices = new int[]{0};
        bSize = 3;
        bSparseComplex = new CooCVectorOld(bSize, bComplexEntries, bIndices);
        expComplexEntries = new CNumber[][]{
                {new CNumber(1).add(bComplexEntries[0]), new CNumber(2.3).add(bComplexEntries[0])},
                {new CNumber(-9), new CNumber(13.5)},
                {new CNumber(9.4), new CNumber(0)}};
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, A.addToEachCol(bSparseComplex));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9.234, -.13)};
        bIndices = new int[]{0};
        bSize = 8;
        bSparseComplex = new CooCVectorOld(bSize, bComplexEntries, bIndices);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(bSparseComplex));
    }
}
