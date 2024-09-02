package org.flag4j.operations_old.dense_sparse.complex;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.operations_old.dense_sparse.coo.complex.ComplexDenseSparseMatrixMultTranspose;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ComplexDenseSparseMatMultTransposeTests {

    static Shape sparseShape;
    static int[][] sparseIndices;
    static int sparseSize;
    static int[] sparseVecIndices;

    static CNumber[][] aEntries;
    static CNumber[] bEntries, bVecEntries, bVecSparseEntries, expEntries;

    static CMatrixOld A;
    static CooCMatrixOld B;
    static CVectorOld bvec;
    static CooCVectorOld bSparse;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[][]{
                {new CNumber(1, 34.3), new CNumber(0.44, -9.4)},
                {new CNumber(85.124, 51), new CNumber(3)},
                {new CNumber(26.24, 160.5), new CNumber(0, -34.5)}};

        bEntries = new CNumber[]{new CNumber(1.334, -5.00024), new CNumber(-73.56, 234.56)};
    }

    static void createMatrices() {
        A = new CMatrixOld(aEntries);
        B = new CooCMatrixOld(sparseShape.swapAxes(0, 1), bEntries, sparseIndices[1], sparseIndices[0]);
    }

    static void createDenseVector() {
        bvec = new CVectorOld(bVecEntries);
    }

    static void createSparseVector() {
        bSparse = new CooCVectorOld(sparseSize, bVecSparseEntries, sparseVecIndices);
    }

    @Test
    void matMatMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        sparseShape = new Shape(2, 5);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 4}};
        createMatrices();
        expEntries = A.mult(new CooCMatrixOld(sparseShape, bEntries, sparseIndices[0], sparseIndices[1])).entries;


        Assertions.assertArrayEquals(expEntries,
                ComplexDenseSparseMatrixMultTranspose.multTranspose(
                        A.entries, A.shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape)
        );
    }
}
