package com.flag4j.operations.dense_sparse.complex;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexDenseCooMatrixMultiplicationTests {

    static Shape sparseShape;
    static int[][] sparseIndices;
    static int sparseSize;
    static int[] sparseVecIndices;

    static CNumber[][] aEntries, expEntries;
    static CNumber[] bEntries, bVecEntries, bVecSparseEntries;

    static CMatrix A;
    static CooCMatrix B;
    static CVector bvec;
    static SparseCVector bSparse;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[][]{
                {new CNumber(1, 34.3), new CNumber(0.44, -9.4)},
                {new CNumber(85.124, 51), new CNumber(3)},
                {new CNumber(26.24, 160.5), new CNumber(0, -34.5)}};

        bEntries = new CNumber[]{new CNumber(1.334, -5.00024), new CNumber(-73.56, 234.56)};
    }

    static void createMatrices() {
        A = new CMatrix(aEntries);
        B = new CooCMatrix(sparseShape, bEntries, sparseIndices[0], sparseIndices[1]);
    }

    static void createDenseVector() {
        bvec = new CVector(bVecEntries);
    }

    static void createSparseVector() {
        bSparse = new SparseCVector(sparseSize, bVecSparseEntries, sparseVecIndices);
    }


    @Test
    void matMatMultTestCase()  {
        // ----------------------- Sub-case 1 -----------------------
        sparseShape = new Shape(2, 5);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 4}};
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("172.842232+40.75596i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("2172.4976+794.6704000000001i")},
                {new CNumber("0.0"), new CNumber("368.567656-357.60642975999997i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-220.68+703.6800000000001i")},
                {new CNumber("0.0"), new CNumber("837.5426799999999+82.9007024i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("8092.32+2537.82i")}};
        createMatrices();

        Assertions.assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.standard(
                        A.entries, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape)
        );
        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.concurrentStandard(
                        A.entries, A.shape,
                        B.entries, B.rowIndices, B.colIndices, B.shape)
        );

        // ----------------------- Sub-case 2 -----------------------
        sparseShape = new Shape(2, 3);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 2}};
        expEntries = new CNumber[][]{
                {new CNumber("368.567656-357.60642975999997i"), new CNumber("4.002000000000001-15.00072i")},
                {new CNumber("-39577.094399999994-5651.525600000002i"), new CNumber("8092.32+2537.82i")}};
        createMatrices();

        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.standard(
                        B.entries, B.rowIndices, B.colIndices, B.shape,
                        A.entries, A.shape)
        );
        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.concurrentStandard(
                        B.entries, B.rowIndices, B.colIndices, B.shape,
                        A.entries, A.shape)
        );
    }


    @Test
    void matVecMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        sparseShape = new Shape(2, 4);
        sparseIndices = new int[][]{
                {0, 1},
                {1, 3}};
        bVecEntries = new CNumber[]{new CNumber("1.334+0.00824i"), new CNumber("324.5+4.1i"),
                new CNumber("-24.5-45.1i"), new CNumber("0.0+6.1255i")};
        expEntries = new CNumber[][]{{new CNumber("453.38398400000005-1617.1084799999999i"),
                new CNumber("-1436.79728-450.59177999999997i")}};
        createMatrices();
        createDenseVector();

        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.standardVector(
                        B.entries, B.rowIndices, B.colIndices, B.shape,
                        bvec.entries, bvec.shape)
        );
        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.concurrentStandardVector(
                        B.entries, B.rowIndices, B.colIndices, B.shape,
                        bvec.entries, bvec.shape)
        );


        // ----------------------- Sub-case 2 -----------------------
        sparseSize = 2;
        sparseVecIndices = new int[]{1};
        bVecSparseEntries = new CNumber[]{new CNumber("1.334+0.00824i")};
        expEntries = new CNumber[][]{
                {new CNumber("0.664416-12.535974400000002i"),
                new CNumber("4.002000000000001+0.024720000000000002i"),
                new CNumber("0.28428000000000003-46.023i")}};
        createMatrices();
        createSparseVector();

        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.standardVector(
                        A.entries, A.shape, bSparse.entries, bSparse.indices
                )
        );
        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.concurrentStandardVector(
                        A.entries, A.shape, bSparse.entries, bSparse.indices
                )
        );
        assertArrayEquals(ArrayUtils.flatten(expEntries),
                ComplexDenseSparseMatrixMultiplication.concurrentBlockedVector(
                        A.entries, A.shape, bSparse.entries, bSparse.indices
                )
        );
    }
}
