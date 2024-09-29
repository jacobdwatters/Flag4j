package org.flag4j.operations.dense_sparse.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseMatrixMultTranspose;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ComplexDenseSparseMatMultTransposeTests {

    static Shape sparseShape;
    static int[][] sparseIndices;
    static int sparseSize;
    static int[] sparseVecIndices;

    static Complex128[][] aEntries;
    static Field<Complex128>[] bEntries, bVecEntries, bVecSparseEntries, expEntries;

    static CMatrix A;
    static CooCMatrix B;
    static CVector bvec;
    static CooCVector bSparse;

    @BeforeAll
    static void setup() {
        aEntries = new Complex128[][]{
                {new Complex128(1, 34.3), new Complex128(0.44, -9.4)},
                {new Complex128(85.124, 51), new Complex128(3)},
                {new Complex128(26.24, 160.5), new Complex128(0, -34.5)}};

        bEntries = new Complex128[]{new Complex128(1.334, -5.00024), new Complex128(-73.56, 234.56)};
    }

    static void createMatrices() {
        A = new CMatrix(aEntries);
        B = new CooCMatrix(sparseShape.swapAxes(0, 1), bEntries, sparseIndices[1], sparseIndices[0]);
    }

    static void createDenseVector() {
        bvec = new CVector(bVecEntries);
    }

    static void createSparseVector() {
        bSparse = new CooCVector(sparseSize, bVecSparseEntries, sparseVecIndices);
    }

    @Test
    void matMatMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        sparseShape = new Shape(2, 5);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 4}};
        createMatrices();
        expEntries = A.mult(new CooCMatrix(sparseShape, bEntries, sparseIndices[0], sparseIndices[1])).entries;


        Assertions.assertArrayEquals(expEntries,
                ComplexDenseSparseMatrixMultTranspose.multTranspose(
                        A.entries, A.shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape)
        );
    }
}
