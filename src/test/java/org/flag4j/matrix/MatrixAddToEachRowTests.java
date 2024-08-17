package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixAddToEachRowTests {
    double[][] aEntries, expEntries;
    double[] bEntries;
    CNumber[][] expComplexEntries;
    CNumber[] bComplexEntries;

    int[] bIndices;
    int bSize;

    MatrixOld A, exp;
    CMatrixOld expComplex;

    VectorOld b;
    CooVector bSparse;
    CVectorOld bComplex;
    CooCVector bSparseComplex;

    @Test
    void vectorTestCase() {
        // ---------------- Sub-case 1  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.7, -8.234};
        b = new VectorOld(bEntries);
        expEntries = new double[][]{
                {1+1.7, 2.3-8.234},
                {-9+1.7, 13.5-8.234},
                {9.4+1.7, 0-8.234}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.7, -8.234, 12.5};
        b = new VectorOld(bEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(b));
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
        bSize = 2;
        bSparse = new CooVector(bSize, bEntries, bIndices);
        expEntries = new double[][]{
                {1, 2.3+3.4},
                {-9, 13.5+3.4},
                {9.4, 0+3.4}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, A.addToEachRow(bSparse));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{3.4};
        bIndices = new int[]{1};
        bSize = 56;
        bSparse = new CooVector(bSize, bEntries, bIndices);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bSparse));
    }


    @Test
    void complexVectorTestCase() {
        // ---------------- Sub-case 1  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9.234, -.13), new CNumber(0, 5.4)};
        bComplex = new CVectorOld(bComplexEntries);
        expComplexEntries = new CNumber[][]{
                {new CNumber(1).add(bComplexEntries[0]), new CNumber(2.3).add(bComplexEntries[1])},
                {new CNumber(-9).add(bComplexEntries[0]), new CNumber(13.5).add(bComplexEntries[1])},
                {new CNumber(9.4).add(bComplexEntries[0]), new CNumber(0).add(bComplexEntries[1])}};
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, A.addToEachRow(bComplex));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9123)};
        bComplex = new CVectorOld(bComplexEntries);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bComplex));
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
        bSize = 2;
        bSparseComplex = new CooCVector(bSize, bComplexEntries, bIndices);
        expComplexEntries = new CNumber[][]{
                {new CNumber(1).add(bComplexEntries[0]), new CNumber(2.3)},
                {new CNumber(-9).add(bComplexEntries[0]), new CNumber(13.5)},
                {new CNumber(9.4).add(bComplexEntries[0]), new CNumber(0)}};
        expComplex = new CMatrixOld(expComplexEntries);

        assertEquals(expComplex, A.addToEachRow(bSparseComplex));

        // ---------------- Sub-case 2  ----------------
        aEntries = new double[][]{
                {1, 2.3},
                {-9, 13.5},
                {9.4, 0}};
        A = new MatrixOld(aEntries);
        bComplexEntries = new CNumber[]{new CNumber(9.234, -.13)};
        bIndices = new int[]{0};
        bSize = 8;
        bSparseComplex = new CooCVector(bSize, bComplexEntries, bIndices);

        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(bSparseComplex));
    }
}
